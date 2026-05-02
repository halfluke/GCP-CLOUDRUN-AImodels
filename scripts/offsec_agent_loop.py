#!/usr/bin/env python3
"""
Local agent loop + parser for models that emit tool calls natively or as text.

Backends:
  - ollama   POST {base}/api/chat
  - openai   POST {base}/v1/chat/completions  (vLLM, OpenAI-compatible servers)

Supported tool output formats:
  - Native: message.tool_calls
  - JSON object: {"name": "...", "arguments": {...}}
  - JSON array: [{"name": "...", "arguments": {...}}]
  - Fenced JSON: ```json ... ```
  - Qwen-style tag: <tool_call>{{...}}</tool_call>

Built-in tools (all filesystem paths must resolve inside ``--base-dir`` / workspace root):
  - list_files(path=".")
  - read_file(path)
  - write_file(path, content, append=False, mkdirs=True)
  - delete_file(path)   [file must resolve inside --base-dir]
  - run_terminal_command(command, cwd=".")   [disabled unless --allow-shell]

Examples:
  python3 scripts/offsec_agent_loop.py \
    --backend ollama \
    --base-url http://127.0.0.1:11434 \
    --model f0rc3ps/nu11secur1tyAIRedTeamLite \
    --prompt "Create ./aaa/suka.txt with content suka, then read it back."

  python3 scripts/offsec_agent_loop.py \
    --backend openai \
    --base-url http://127.0.0.1:8080 \
    --model DeepHat/DeepHat-V1-7B \
    --prompt "Run pwd with run_terminal_command."

  streamlit run scripts/offsec_streamlit_app.py
  # requires: pip install -r requirements-streamlit.txt
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from urllib import error, request

Backend = Literal["ollama", "openai"]


class AgentObserver(Protocol):
    """Optional UI sink for :func:`run_agent_steps` (default is terminal prints)."""

    def info(self, msg: str) -> None: ...
    def assistant_turn(self, step: int, max_steps: int, content: str) -> None: ...
    def tool_call(
        self, name: str, arguments: dict[str, Any], result: dict[str, Any]
    ) -> None: ...
    def truncated_in_history(self) -> None: ...


class PrintObserver:
    """Default observer: mirror historical ``print`` behaviour."""

    def info(self, msg: str) -> None:
        print(msg, flush=True)

    def assistant_turn(self, step: int, max_steps: int, content: str) -> None:
        print(f"\n--- step {step} assistant ---", flush=True)
        print(content if content else "(empty content)", flush=True)

    def tool_call(
        self, name: str, arguments: dict[str, Any], result: dict[str, Any]
    ) -> None:
        print(
            f"\n[tool] {name}({json.dumps(arguments, ensure_ascii=False)})",
            flush=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    def truncated_in_history(self) -> None:
        print("  (truncated in chat history sent back to model)", flush=True)


_DEFAULT_PRINT_OBSERVER = PrintObserver()


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", re.IGNORECASE)
JSON_ARRAY_SNIPPET_RE = re.compile(r"(\[\s*\{[\s\S]*\}\s*\])")
JSON_OBJECT_SNIPPET_RE = re.compile(r"(\{\s*\"name\"[\s\S]*\})")
OLLAMA_MODEL_NOT_FOUND_RE = re.compile(r"model\s+'([^']+)'\s+not\s+found", re.IGNORECASE)

# When --system is omitted, balance tool use vs normal chat and avoid dumping raw tool JSON.
DEFAULT_SYSTEM = (
    "You are a helpful agent with tools for the user's workspace. "
    "For greetings, small talk, or questions that do not need files or shell output, "
    "answer in plain language and do not call tools. "
    "When the user clearly asks to list, read, create, write, or delete files or run a command, "
    "use the appropriate tool. "
    "When you need a tool, emit a single JSON object with \"name\" and \"arguments\" "
    "(or use native tool_calls if supported); avoid mixing long prose into that JSON block. "
    "After a tool returns, reply briefly in natural language; never paste the raw tool-result JSON "
    "back as your entire message. "
    "Do not repeat the same tool call unless the user asks again."
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


def http_post_json(
    url: str,
    payload: dict[str, Any],
    *,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    data = json.dumps(payload).encode()
    req = request.Request(
        url,
        data=data,
        method="POST",
        headers=headers,
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())
    except error.HTTPError as e:
        body = e.read().decode(errors="replace")
        hint = ""
        m = OLLAMA_MODEL_NOT_FOUND_RE.search(body)
        if e.code == 404 and m:
            hint = f"\n(Hint: install locally: ollama pull {m.group(1)})"
        if e.code == 400 and "context length" in body.lower():
            hint = (
                "\n(Hint: prompt + tools exceed server context — lower --max-tokens, "
                "match --context-limit to your vLLM max_model_len, shrink dirs via "
                "--tool-list-cap / --tool-chars-cap, or start a fresh interactive session.)"
            )
        raise RuntimeError(f"HTTP {e.code}: {body}{hint}") from e


def unwrap_double_braces(s: str) -> str:
    t = s.strip()
    while t.startswith("{{") and t.endswith("}}"):
        t = t[1:-1].strip()
    return t


def normalize_parsed_calls(parsed: Any) -> list[ToolCall]:
    if isinstance(parsed, dict) and "name" in parsed:
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []
    out: list[ToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        args = item.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        out.append(ToolCall(name=name, arguments=args))
    return out


def dedupe_calls(calls: list[ToolCall]) -> list[ToolCall]:
    seen: set[tuple[str, str]] = set()
    out: list[ToolCall] = []
    for c in calls:
        key = (c.name, json.dumps(c.arguments, sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def calls_signature(calls: list[ToolCall]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((c.name, json.dumps(c.arguments, sort_keys=True, ensure_ascii=False)) for c in calls))


def extract_balanced_segments(text: str, open_ch: str, close_ch: str) -> list[str]:
    """Non-overlapping balanced segments starting at top-level open_ch (string-aware)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != open_ch:
            i += 1
            continue
        depth = 0
        start = i
        j = i
        in_string = False
        escape = False
        quote_char = ""
        while j < n:
            c = text[j]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == quote_char:
                    in_string = False
                j += 1
                continue
            if c in "\"'":
                in_string = True
                quote_char = c
                j += 1
                continue
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    out.append(text[start : j + 1])
                    i = j + 1
                    break
            j += 1
        else:
            i += 1
    return out


def parse_calls_from_concatenated_json(text: str) -> list[ToolCall]:
    """Handle [...][...], prose + [...], and lines containing multiple JSON arrays."""
    text = re.sub(r"\]\s*\\\s*\[", "][", text)
    merged: list[ToolCall] = []
    for seg in extract_balanced_segments(text, "[", "]"):
        try:
            parsed = json.loads(seg.strip())
        except json.JSONDecodeError:
            continue
        merged.extend(normalize_parsed_calls(parsed))
    if not merged:
        for seg in extract_balanced_segments(text, "{", "}"):
            if '"name"' not in seg:
                continue
            try:
                parsed = json.loads(seg.strip())
            except json.JSONDecodeError:
                continue
            merged.extend(normalize_parsed_calls(parsed))
    return dedupe_calls(merged)


def chat_endpoint(base_url: str, backend: Backend) -> str:
    base = base_url.rstrip("/")
    if backend == "ollama":
        return f"{base}/api/chat"
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def normalize_chat_response(backend: Backend, raw: dict[str, Any]) -> dict[str, Any]:
    """Always return a dict with key \"message\" like Ollama's /api/chat."""
    if backend == "ollama":
        return raw if isinstance(raw.get("message"), dict) else {**raw, "message": {}}
    err = raw.get("error")
    if isinstance(err, str):
        raise RuntimeError(f"API error: {err}")
    if isinstance(err, dict):
        msg = err.get("message", json.dumps(err))
        raise RuntimeError(f"API error: {msg}")
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"message": {}}
    ch0 = choices[0]
    if not isinstance(ch0, dict):
        return {"message": {}}
    msg = ch0.get("message")
    if not isinstance(msg, dict):
        return {"message": {}}
    return {"message": msg}


def strip_chat_template_leaks(text: str) -> str:
    """Remove common ChatML/special tokens leaked into assistant content before JSON parse."""
    t = text.strip()
    # Repeat: <|im_start|> may appear multiple times or with a role on the same line.
    for _ in range(4):
        t2 = re.sub(r"^<\|im_start\|>\s*", "", t, count=1, flags=re.MULTILINE)
        if t2 == t:
            break
        t = t2.strip()
    t = re.sub(r"^<\|im_start\|>[^\n]*\n", "", t, count=1)
    t = re.sub(r"^(assistant|user|system)\s*\n", "", t.strip(), count=1, flags=re.IGNORECASE)
    t = re.sub(r"<\|im_end\|>\s*$", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"<\|redacted_im_end\|>\s*$", "", t.strip(), flags=re.IGNORECASE)
    return t.strip()


def parse_tool_calls_from_text(content: str | None) -> list[ToolCall]:
    if not content:
        return []
    text = strip_chat_template_leaks(content)

    # 1) <tool_call>...</tool_call>
    tagged = TOOL_CALL_TAG_RE.findall(text)
    if tagged:
        calls: list[ToolCall] = []
        for block in tagged:
            inner = unwrap_double_braces(block)
            try:
                parsed = json.loads(inner)
            except json.JSONDecodeError:
                continue
            calls.extend(normalize_parsed_calls(parsed))
        if calls:
            return dedupe_calls(calls)

    # 2) fenced json
    m = JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    # 3) direct json object/array
    try:
        parsed = json.loads(text)
        calls = normalize_parsed_calls(parsed)
        if calls:
            return dedupe_calls(calls)
    except json.JSONDecodeError:
        pass

    # 4) snippet recovery
    for rx in (JSON_ARRAY_SNIPPET_RE, JSON_OBJECT_SNIPPET_RE):
        m2 = rx.search(text)
        if not m2:
            continue
        snippet = unwrap_double_braces(m2.group(1))
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        calls = normalize_parsed_calls(parsed)
        if calls:
            return dedupe_calls(calls)
    # 5) Multiple concatenated arrays / objects (models often emit [...][...])
    glued = parse_calls_from_concatenated_json(text)
    if glued:
        return glued
    return []


def path_must_be_under_base(p: Path, base_r: Path) -> dict[str, Any] | None:
    """Return an error dict if *p* is not under *base_r* (each path resolved)."""
    try:
        p.resolve().relative_to(base_r.resolve())
    except ValueError:
        return {"ok": False, "error": f"path outside workspace (--base-dir): {p}"}
    return None


def resolve_workspace_path(
    base_dir: Path, raw: str
) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolve *raw* under workspace rules; fail if the result leaves ``base_dir``."""
    base_r = base_dir.resolve()
    s = (raw or ".").strip() or "."
    if s.startswith("~/"):
        p = Path(os.path.expanduser(s)).resolve()
    elif s.startswith("/"):
        p = Path(s).resolve()
    elif s.startswith("home/"):
        p = Path("/" + s).resolve()
    else:
        p = (base_dir / s).resolve()
    err = path_must_be_under_base(p, base_r)
    if err:
        return None, err
    return p, None


_SHELL_SEGMENT_SPLIT = re.compile(r"\s*(?:&&|\|\||;|\n)\s*")


def check_shell_cd_stays_in_workspace(
    command: str, cwd: Path, base_dir: Path
) -> dict[str, Any] | None:
    """Reject if a top-level ``cd`` in the command would leave the workspace.

    Best-effort only: subshells, ``eval``, heredocs, and arbitrary binaries can
    still escape — use ``--allow-shell`` only when you accept that residual risk.
    """
    base_r = base_dir.resolve()
    cwd_err = path_must_be_under_base(cwd, base_r)
    if cwd_err:
        return cwd_err
    cur = cwd.resolve()
    cd_re = re.compile(r"^\s*cd\s+(.+?)\s*$", re.IGNORECASE)
    for seg in _SHELL_SEGMENT_SPLIT.split(command):
        seg = seg.strip()
        if not seg or seg.startswith("#"):
            continue
        m = cd_re.match(seg)
        if not m:
            continue
        raw_target = m.group(1).strip().strip("'\"")
        if raw_target in {"", ".", "-", "$OLDPWD"}:
            continue
        if raw_target.startswith("~/"):
            nxt = Path(os.path.expanduser(raw_target)).resolve()
        elif raw_target.startswith("/"):
            nxt = Path(raw_target).resolve()
        else:
            nxt = (cur / raw_target).resolve()
        err = path_must_be_under_base(nxt, base_r)
        if err:
            return {
                "ok": False,
                "error": (
                    f"refusing shell: cd to {raw_target!r} resolves to {nxt}, "
                    f"outside workspace {base_r}"
                ),
            }
        cur = nxt
    return None


def shrink_tool_payload_for_llm(
    result: dict[str, Any],
    tool_name: str,
    *,
    max_list_entries: int,
    max_tool_chars: int,
) -> dict[str, Any]:
    """Reduce payload size embedded in chat history (full output still printed to terminal)."""
    if not isinstance(result, dict):
        return result
    if tool_name == "list_files" and result.get("ok") and isinstance(result.get("entries"), list):
        entries = result["entries"]
        if len(entries) > max_list_entries:
            return {
                **result,
                "entries": entries[:max_list_entries],
                "entries_truncated": True,
                "entries_total": len(entries),
            }
        return result
    if tool_name == "read_file" and result.get("ok") and isinstance(result.get("content"), str):
        c = result["content"]
        if len(c) > max_tool_chars:
            return {
                **result,
                "content": c[:max_tool_chars] + "\n… [truncated]",
                "content_truncated": True,
            }
        return result
    if tool_name in {"run_terminal_command", "run_command", "shell"}:
        out = dict(result)
        changed = False
        for key in ("stdout", "stderr"):
            v = out.get(key)
            if isinstance(v, str) and len(v) > max_tool_chars:
                out[key] = v[:max_tool_chars] + "\n… [truncated]"
                out[f"{key}_truncated"] = True
                changed = True
        return out if changed else result
    return result


def rough_prompt_chars(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    n = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    if tools:
        n += len(json.dumps(tools, ensure_ascii=False))
    return n


def capped_output_tokens(
    requested: int,
    *,
    context_limit: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    """Keep input + requested output under context_limit using a crude char-based estimate."""
    if requested <= 0:
        return 0
    chars = rough_prompt_chars(messages, tools)
    est_in = max(chars // 3, chars // 6)
    reserve = 64
    room = context_limit - est_in - reserve
    return min(requested, max(24, room))


def tool_list_files(args: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    p, err = resolve_workspace_path(base_dir, str(args.get("path", ".")))
    if err:
        return err
    assert p is not None
    if not p.exists():
        return {"ok": False, "error": f"path does not exist: {p}"}
    if not p.is_dir():
        return {"ok": False, "error": f"not a directory: {p}"}
    entries = sorted(x.name for x in p.iterdir())
    return {"ok": True, "path": str(p), "entries": entries}


def tool_read_file(args: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    raw = args.get("path")
    if not isinstance(raw, str):
        return {"ok": False, "error": "path is required (string)"}
    p, err = resolve_workspace_path(base_dir, raw)
    if err:
        return err
    assert p is not None
    if not p.exists():
        return {"ok": False, "error": f"file does not exist: {p}"}
    if p.is_dir():
        return {"ok": False, "error": f"path is a directory: {p}"}
    try:
        content = p.read_text()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"read failed: {e}"}
    return {"ok": True, "path": str(p), "content": content}


def tool_write_file(args: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    raw = args.get("path")
    content = args.get("content", "")
    append = bool(args.get("append", False))
    mkdirs = bool(args.get("mkdirs", True))
    if not isinstance(raw, str):
        return {"ok": False, "error": "path is required (string)"}
    if not isinstance(content, str):
        return {"ok": False, "error": "content must be string"}
    p, err = resolve_workspace_path(base_dir, raw)
    if err:
        return err
    assert p is not None
    try:
        if mkdirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with p.open(mode, encoding="utf-8") as f:
            f.write(content)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"write failed: {e}"}
    return {"ok": True, "path": str(p), "bytes": len(content), "append": append}


def tool_delete_file(args: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    raw = args.get("path")
    if not isinstance(raw, str):
        return {"ok": False, "error": "path is required (string)"}
    p, err = resolve_workspace_path(base_dir, raw)
    if err:
        return err
    assert p is not None
    if not p.exists():
        return {"ok": False, "error": f"path does not exist: {p}"}
    if p.is_dir():
        return {"ok": False, "error": f"path is a directory (not deleting): {p}"}
    try:
        p.unlink()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"delete failed: {e}"}
    return {"ok": True, "path": str(p), "deleted": True}


def tool_run_shell(args: dict[str, Any], base_dir: Path, allow_shell: bool) -> dict[str, Any]:
    if not allow_shell:
        return {"ok": False, "error": "run_terminal_command disabled; use --allow-shell"}
    cmd = args.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return {"ok": False, "error": "command is required (string)"}
    cwd_raw = args.get("cwd", ".")
    cwd, err = resolve_workspace_path(base_dir, str(cwd_raw))
    if err:
        return err
    assert cwd is not None
    cd_err = check_shell_cd_stays_in_workspace(cmd, cwd, base_dir)
    if cd_err:
        return cd_err
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=str(cwd),
            shell=True,  # noqa: S602
            text=True,
            capture_output=True,
            timeout=120,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "cwd": str(cwd),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"command failed: {e}"}


def execute_tool(call: ToolCall, base_dir: Path, allow_shell: bool) -> dict[str, Any]:
    name = call.name
    args = call.arguments
    if name == "list_files":
        return tool_list_files(args, base_dir)
    if name == "read_file":
        return tool_read_file(args, base_dir)
    if name == "write_file":
        return tool_write_file(args, base_dir)
    if name in {"delete_file", "remove_file", "unlink"}:
        return tool_delete_file(args, base_dir)
    if name in {"run_terminal_command", "run_command", "shell"}:
        return tool_run_shell(args, base_dir, allow_shell)
    return {"ok": False, "error": f"unknown tool: {name}"}


def extract_calls_from_response(resp: dict[str, Any]) -> list[ToolCall]:
    msg = ((resp.get("message") or {}) if isinstance(resp, dict) else {})
    # Native tool_calls first.
    native = msg.get("tool_calls")
    if isinstance(native, list) and native:
        out: list[ToolCall] = []
        for tc in native:
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if isinstance(name, str) and isinstance(args, dict):
                tid = tc.get("id")
                cid = tid if isinstance(tid, str) else None
                out.append(ToolCall(name=name, arguments=args, call_id=cid))
        if out:
            return dedupe_calls(out)
    # Fallback parse from content.
    return dedupe_calls(parse_tool_calls_from_text(msg.get("content")))


def build_tools(*, allow_shell: bool = True) -> list[dict[str, Any]]:
    """OpenAI-style tool definitions sent to the model.

    When ``allow_shell`` is false, ``run_terminal_command`` is omitted so the model
    is not prompted with a tool it cannot run.
    """
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List directory entries. Path must stay inside the workspace root (no .. escapes).",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file; path must resolve inside the workspace root.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write file content; path must resolve inside the workspace root.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "append": {"type": "boolean"},
                        "mkdirs": {"type": "boolean"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete a file under the workspace only (--base-dir). Refuses directories.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
    ]
    if allow_shell:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "run_terminal_command",
                    "description": (
                        "Run shell with cwd inside the workspace. Top-level cd that leaves "
                        "the workspace is rejected; this is not a full sandbox (e.g. cat /etc/passwd may work)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "cwd": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            }
        )
    return tools


def run_agent_steps(
    *,
    backend: Backend,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_steps: int,
    base_dir: Path,
    allow_shell: bool,
    api_key: str | None,
    max_tokens: int,
    temperature: float | None,
    context_limit: int,
    max_list_entries: int,
    max_tool_chars: int,
    observer: AgentObserver | None = None,
) -> int:
    obs = observer if observer is not None else _DEFAULT_PRINT_OBSERVER
    extra_headers: dict[str, str] | None = None
    if api_key:
        extra_headers = {"Authorization": f"Bearer {api_key}"}

    prev_sig: tuple[tuple[str, str], ...] | None = None
    for step in range(1, max_steps + 1):
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        eff_mt = (
            capped_output_tokens(
                max_tokens,
                context_limit=context_limit,
                messages=messages,
                tools=tools or [],
            )
            if max_tokens > 0
            else 0
        )
        if max_tokens > 0 and eff_mt < max_tokens:
            obs.info(
                f"  (max_tokens {max_tokens} → {eff_mt} for ~{context_limit} context budget)"
            )
        if max_tokens > 0 and eff_mt <= 48:
            obs.info(
                "  (warning: tight context — shorten chat, list smaller dirs, or raise server max_model_len)"
            )

        if backend == "openai":
            payload["tool_choice"] = "auto"
            if eff_mt > 0:
                payload["max_tokens"] = eff_mt
            if temperature is not None:
                payload["temperature"] = temperature
        else:
            opts: dict[str, Any] = {}
            if eff_mt > 0:
                opts["num_predict"] = eff_mt
            if temperature is not None:
                opts["temperature"] = temperature
            if opts:
                payload["options"] = opts

        # Tool output prints right after each tool runs; this POST asks for the next assistant
        # message (prose or more tools). Without a token cap, vLLM may generate for a long time.
        cap = f", gen_cap≈{eff_mt}" if eff_mt > 0 else ""
        obs.info(f"\n→ calling API (inner step {step}/{max_steps}{cap}) …")
        raw = http_post_json(url, payload, extra_headers=extra_headers)
        resp = normalize_chat_response(backend, raw)
        msg = resp.get("message", {})
        raw_c = msg.get("content")
        text_out = raw_c if isinstance(raw_c, str) else ""
        obs.assistant_turn(step, max_steps, text_out)

        calls = extract_calls_from_response(resp)
        if not calls:
            if text_out.strip():
                messages.append({"role": "assistant", "content": text_out})
            obs.info("\n(no tool calls detected; done)")
            return 0

        sig = calls_signature(calls)
        if prev_sig is not None and sig == prev_sig:
            obs.info(
                "\n(stopping: model repeated the same tool batch; "
                "say 'summarize only' or raise --max-steps if needed)"
            )
            return 0

        call_ids = [c.call_id or f"call_s{step}_{i}" for i, c in enumerate(calls)]

        tc_payload: list[dict[str, Any]] = []
        for idx, c in enumerate(calls):
            tid = call_ids[idx]
            if backend == "openai":
                tc_payload.append(
                    {
                        "id": tid,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": json.dumps(c.arguments, ensure_ascii=False),
                        },
                    }
                )
            else:
                tc_payload.append(
                    {
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": c.arguments,
                        },
                    }
                )

        # Ollama: empty assistant content avoids `{`/`}` in history breaking templates.
        # OpenAI: null content is typical when tool_calls are present.
        if backend == "openai":
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": tc_payload,
            }
        else:
            assistant_msg = {"role": "assistant", "content": "", "tool_calls": tc_payload}
        messages.append(assistant_msg)

        for c, tid in zip(calls, call_ids, strict=True):
            result = execute_tool(c, base_dir, allow_shell)
            obs.tool_call(c.name, c.arguments, result)
            payload_result = shrink_tool_payload_for_llm(
                result,
                c.name,
                max_list_entries=max_list_entries,
                max_tool_chars=max_tool_chars,
            )
            if payload_result != result:
                obs.truncated_in_history()
            tool_body = json.dumps(payload_result, ensure_ascii=False)
            if backend == "openai":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tid,
                        "content": tool_body,
                    }
                )
            else:
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": c.name,
                        "content": tool_body,
                    }
                )

        prev_sig = sig

    obs.info("\n(max steps reached for this turn)")
    return 0


def append_initial_system(messages: list[dict[str, Any]], system: str | None) -> None:
    """If system is set, use it. If omitted (None), inject DEFAULT_SYSTEM. Empty string adds none."""
    if system:
        messages.append({"role": "system", "content": system})
    elif system is None:
        messages.append({"role": "system", "content": DEFAULT_SYSTEM})


def run_single_prompt(
    *,
    backend: Backend,
    base_url: str,
    model: str,
    system: str | None,
    prompt: str,
    max_steps: int,
    base_dir: Path,
    allow_shell: bool,
    api_key: str | None,
    max_tokens: int,
    temperature: float | None,
    context_limit: int,
    max_list_entries: int,
    max_tool_chars: int,
) -> int:
    messages: list[dict[str, Any]] = []
    append_initial_system(messages, system)
    messages.append({"role": "user", "content": prompt})
    tools = build_tools(allow_shell=allow_shell)
    url = chat_endpoint(base_url, backend)
    return run_agent_steps(
        backend=backend,
        url=url,
        model=model,
        messages=messages,
        tools=tools,
        max_steps=max_steps,
        base_dir=base_dir,
        allow_shell=allow_shell,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        context_limit=context_limit,
        max_list_entries=max_list_entries,
        max_tool_chars=max_tool_chars,
    )


def run_interactive(
    *,
    backend: Backend,
    base_url: str,
    model: str,
    system: str | None,
    max_steps: int,
    base_dir: Path,
    allow_shell: bool,
    api_key: str | None,
    max_tokens: int,
    temperature: float | None,
    context_limit: int,
    max_list_entries: int,
    max_tool_chars: int,
) -> int:
    messages: list[dict[str, Any]] = []
    append_initial_system(messages, system)
    tools = build_tools(allow_shell=allow_shell)
    url = chat_endpoint(base_url, backend)
    print("Interactive mode ready. Type /exit (or /quit) to stop.")
    while True:
        try:
            user_input = input("\nYou> ").strip()
        except EOFError:
            print("\nEOF received, exiting.")
            return 0
        except KeyboardInterrupt:
            print("\nInterrupted, exiting.")
            return 0
        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit"}:
            print("Bye.")
            return 0
        messages.append({"role": "user", "content": user_input})
        rc = run_agent_steps(
            backend=backend,
            url=url,
            model=model,
            messages=messages,
            tools=tools,
            max_steps=max_steps,
            base_dir=base_dir,
            allow_shell=allow_shell,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            context_limit=context_limit,
            max_list_entries=max_list_entries,
            max_tool_chars=max_tool_chars,
        )
        if rc != 0:
            return rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agent loop + parser (Ollama /api/chat or OpenAI-compatible /v1/chat/completions)"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="ollama=/api/chat, openai=/v1/chat/completions (e.g. vLLM)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Server base URL (no path): Ollama default 11434, vLLM proxy often 8080",
    )
    parser.add_argument("--model", required=True, help="Model name/id for the server")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Bearer token for OpenAI-compatible APIs",
    )
    parser.add_argument("--prompt", default=None, help="User prompt (not needed with --interactive)")
    parser.add_argument(
        "--system",
        default=None,
        help="System prompt; omit to use a built-in workspace/tools-oriented default. Use \"\" for none.",
    )
    parser.add_argument("--max-steps", type=int, default=8, help="Max loop steps")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Cap generation per request (OpenAI: max_tokens; Ollama: options.num_predict). "
        "0 = omit (server default; vLLM may run very long).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; omit for server default.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=4096,
        help="Server context window for budgeting input+output (match vLLM --max-model-len / Ollama num_ctx).",
    )
    parser.add_argument(
        "--tool-list-cap",
        type=int,
        default=40,
        help="Max directory entries kept when sending list_files results back to the model.",
    )
    parser.add_argument(
        "--tool-chars-cap",
        type=int,
        default=3500,
        help="Max characters per read_file body or shell stdout/stderr in chat history.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep chat open for multiple user turns",
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Workspace base directory used for relative paths",
    )
    parser.add_argument(
        "--allow-shell",
        action="store_true",
        help="Enable run_terminal_command tool",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    backend = cast(Backend, args.backend)
    if args.interactive:
        return run_interactive(
            backend=backend,
            base_url=args.base_url,
            model=args.model,
            system=args.system,
            max_steps=args.max_steps,
            base_dir=base_dir,
            allow_shell=args.allow_shell,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            context_limit=args.context_limit,
            max_list_entries=args.tool_list_cap,
            max_tool_chars=args.tool_chars_cap,
        )
    if not args.prompt:
        print("Error: provide --prompt for one-shot mode, or use --interactive.")
        return 2
    return run_single_prompt(
        backend=backend,
        base_url=args.base_url,
        model=args.model,
        system=args.system,
        prompt=args.prompt,
        max_steps=args.max_steps,
        base_dir=base_dir,
        allow_shell=args.allow_shell,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        context_limit=args.context_limit,
        max_list_entries=args.tool_list_cap,
        max_tool_chars=args.tool_chars_cap,
    )


if __name__ == "__main__":
    sys.exit(main())
