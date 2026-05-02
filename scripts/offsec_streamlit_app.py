#!/usr/bin/env python3
"""Streamlit UI for ``offsec_agent_loop`` (workspace tools + vLLM/Ollama).

  pip install -r requirements-streamlit.txt
  streamlit run scripts/offsec_streamlit_app.py

Use **Reset chat** after changing model or base URL so history matches the backend.
"""

from __future__ import annotations

import json
import ssl
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import streamlit as st

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from offsec_agent_loop import (  # noqa: E402
    Backend,
    DEFAULT_SYSTEM,
    build_tools,
    chat_endpoint,
    run_agent_steps,
)

# OpenAI-compatible APIs commonly accept 0–2; local servers may ignore or clamp.
TEMP_MIN = 0.0
TEMP_MAX = 2.0
TEMP_STEP = 0.05

STREAMLIT_DEFAULT_TOOL_DISCIPLINE = (
    "When describing tool use or results: do not invent commands, paths, or outputs. "
    "Base answers strictly on actual tool results. If a tool failed or returned nothing, "
    "say so plainly. Do not add plausible-sounding filler or hallucinated file contents."
)

MODEL_FETCH_TIMEOUT_S = 90
MODEL_FETCH_ATTEMPTS = 5
MODEL_FETCH_RETRY_PAUSE_S = 4.0


def _ssl_context(insecure: bool) -> ssl.SSLContext | None:
    if not insecure:
        return None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_json_get(
    url: str,
    *,
    bearer: str | None,
    timeout: float,
    insecure_tls: bool,
) -> Any:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json, */*;q=0.8")
    if bearer:
        req.add_header("Authorization", f"Bearer {bearer}")
    ctx = _ssl_context(insecure_tls) if url.lower().startswith("https:") else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read().decode()
    return json.loads(raw)


def _transient_models_fetch_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (502, 503, 504)
    if isinstance(
        exc,
        (urllib.error.URLError, TimeoutError, OSError, ConnectionError, BrokenPipeError),
    ):
        return True
    return False


def _http_json_get_resilient(
    url: str,
    *,
    bearer: str | None,
    timeout: float,
    insecure_tls: bool,
) -> Any:
    """Retry on connection failures and typical «instance starting» HTTP statuses (serverless warm-up)."""
    last_exc: BaseException | None = None
    for attempt in range(MODEL_FETCH_ATTEMPTS):
        try:
            return _http_json_get(
                url, bearer=bearer, timeout=timeout, insecure_tls=insecure_tls
            )
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            ValueError,
        ):
            raise
        except Exception as exc:
            last_exc = exc
            if not _transient_models_fetch_error(exc):
                raise
            if attempt + 1 >= MODEL_FETCH_ATTEMPTS:
                raise
            time.sleep(MODEL_FETCH_RETRY_PAUSE_S)
    assert last_exc is not None
    raise last_exc


def _format_models_error(url: str, exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode(errors="replace")[:800]
        except Exception:
            pass
        return (
            f"{type(exc).__name__}: HTTP {exc.code} {exc.reason!s}\n"
            f"URL: {url}\n"
            f"{body}"
        ).strip()
    if isinstance(exc, urllib.error.URLError):
        return f"{type(exc).__name__}: {exc.reason!s}\nURL: {url}"
    return f"{type(exc).__name__}: {exc}\nURL: {url}"


def _parse_openai_models_payload(data: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for item in data.get("data") or []:
        if not isinstance(item, dict):
            continue
        mid = item.get("id") or item.get("model")
        if mid:
            ids.append(str(mid))
    return ids


def _openai_model_url_candidates(root: str) -> list[str]:
    """Avoid ``/v1/v1/models`` when base URL already ends with ``/v1``."""
    r = root.strip().rstrip("/")
    if not r:
        return []
    cand: list[str] = []
    if r.endswith("/v1"):
        cand.append(f"{r}/models")
        base = r[:-3].rstrip("/")
        if base:
            cand.append(f"{base}/v1/models")
    else:
        cand.append(f"{r}/v1/models")
        cand.append(f"{r}/models")
    seen: set[str] = set()
    out: list[str] = []
    for u in cand:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _fetch_models_list(
    backend: Backend,
    base_url: str,
    api_key: str,
    *,
    insecure_tls: bool,
) -> tuple[list[str], str | None]:
    root = base_url.strip().rstrip("/")
    if not root:
        return [], "Base URL is empty."
    bearer = api_key.strip() or None
    if backend == "openai":
        last_hint: str | None = None
        for url in _openai_model_url_candidates(root):
            try:
                data = _http_json_get_resilient(
                    url,
                    bearer=bearer,
                    timeout=MODEL_FETCH_TIMEOUT_S,
                    insecure_tls=insecure_tls,
                )
            except urllib.error.HTTPError as e:
                last_hint = _format_models_error(url, e)
                if e.code in (404, 405, 410):
                    continue
                return [], last_hint
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                UnicodeDecodeError,
                ValueError,
                OSError,
            ) as e:
                last_hint = _format_models_error(url, e)
                continue
            if not isinstance(data, dict):
                last_hint = f"Expected JSON object from {url}, got {type(data).__name__}."
                continue
            ids = _parse_openai_models_payload(data)
            if ids:
                return sorted(set(ids)), None
            last_hint = (
                f"HTTP 200 from {url} but no model ids parsed "
                f"(top-level JSON keys: {list(data.keys())!r})."
            )
        return [], last_hint or "Could not reach an OpenAI-style /v1/models endpoint."
    url = f"{root}/api/tags"
    try:
        data = _http_json_get_resilient(
            url,
            bearer=None,
            timeout=MODEL_FETCH_TIMEOUT_S,
            insecure_tls=insecure_tls,
        )
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        ValueError,
        OSError,
    ) as e:
        return [], _format_models_error(url, e)
    if not isinstance(data, dict):
        return [], f"Expected JSON object from {url}, got {type(data).__name__}."
    names: list[str] = []
    for item in data.get("models") or []:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    if not names:
        return [], (
            f"HTTP 200 from {url} but no models in JSON "
            f"(top-level keys: {list(data.keys())!r}). "
            "Is Backend set to **ollama** and Base URL the Ollama host (e.g. :11434)?"
        )
    return sorted(set(names)), None


def _inject_workspace_root(messages: list[dict[str, Any]], root: Path) -> None:
    """Make tool root unambiguous (`.` == Streamlit cwd is easy to confuse in prose)."""
    note = (
        f"TOOL WORKSPACE ROOT (authoritative): {root}\n"
        "All relative paths for list_files, read_file, write_file, delete_file, and the "
        "default cwd for run_terminal_command are resolved under this directory only.\n"
        "When the user says “current directory” in this app, they mean this root unless "
        "they give a different absolute path.\n"
        "Do not assume other directories (e.g. where a server process was started) unless "
        "the user explicitly names them.\n"
        "File tools cannot read or write outside this root. Shell uses the same cwd rules; "
        "top-level `cd` leaving the root is rejected, but shell is still not a full sandbox."
    )
    for m in messages:
        if m.get("role") == "system":
            prev = (m.get("content") or "").strip()
            m["content"] = (prev + "\n\n" + note).strip() if prev else note
            return
    messages.insert(0, {"role": "system", "content": note})


class CollectObserver:
    """Gather UI rows for one ``run_agent_steps`` invocation."""

    def __init__(self) -> None:
        self.rows: list[tuple[Any, ...]] = []
        self.log_chunks: list[str] = []

    def info(self, msg: str) -> None:
        self.log_chunks.append(msg.rstrip())

    def assistant_turn(self, step: int, max_steps: int, content: str) -> None:
        self.rows.append(("assistant", step, max_steps, content))

    def tool_call(
        self, name: str, arguments: dict[str, Any], result: dict[str, Any]
    ) -> None:
        self.rows.append(("tool", name, arguments, result))

    def truncated_in_history(self) -> None:
        self.rows.append(("trunc",))


def _init_session(system_text: str, workspace_root: Path) -> None:
    st.session_state.agent_messages = []
    st.session_state.chat_ui = []
    extra = system_text.strip()
    if not extra:
        extra = STREAMLIT_DEFAULT_TOOL_DISCIPLINE
    st.session_state.agent_messages.append(
        {"role": "system", "content": f"{DEFAULT_SYSTEM}\n\n{extra}"}
    )
    _inject_workspace_root(st.session_state.agent_messages, workspace_root)
    st.session_state["workspace_root"] = str(workspace_root.resolve())


def main() -> None:
    st.set_page_config(page_title="Offsec agent", layout="wide")
    if "temperature_value" not in st.session_state:
        st.session_state.temperature_value = 0.2
    st.title("Offsec agent loop")
    st.caption(
        "Local tools + Ollama or OpenAI-compatible / vLLM — same core as "
        "`scripts/offsec_agent_loop.py`."
    )

    with st.sidebar:
        st.header("Connection")
        st.caption(
            "Serverless backends (e.g. Cloud Run) can take a minute or two to cold-start. "
            "Model listing uses a long timeout and retries; use **Refresh models** if needed."
        )
        backend_sel = st.selectbox("Backend", ["openai", "ollama"], index=0)
        backend = cast(Backend, backend_sel)
        base_url = st.text_input(
            "Base URL (no path)",
            value="http://127.0.0.1:8080" if backend == "openai" else "http://127.0.0.1:11434",
        )
        api_key = st.text_input("API key (Bearer)", value="", type="password")
        insecure_tls = st.checkbox(
            "Skip TLS certificate verify (dev / self-signed HTTPS only)",
            value=False,
            key="models_insecure_tls",
        )
        _mk = (backend, base_url.strip(), api_key.strip(), insecure_tls)
        if st.session_state.get("_models_cache_key") != _mk:
            st.session_state["_models_cache_key"] = _mk
            mlist, mdia = _fetch_models_list(
                backend,
                base_url,
                api_key,
                insecure_tls=insecure_tls,
            )
            st.session_state["_models_list"] = mlist
            st.session_state["_models_fetch_diag"] = mdia
        models_list = list(st.session_state.get("_models_list") or [])
        _models_diag = st.session_state.get("_models_fetch_diag")
        row_m = st.columns([2, 1])
        with row_m[0]:
            if models_list:
                st.caption(f"{len(models_list)} model(s) from API")
            else:
                st.caption("Could not list models — enter id manually")
        with row_m[1]:
            if st.button("Refresh models"):
                mlist, mdia = _fetch_models_list(
                    backend,
                    base_url,
                    api_key,
                    insecure_tls=insecure_tls,
                )
                st.session_state["_models_list"] = mlist
                st.session_state["_models_fetch_diag"] = mdia
                st.session_state["_models_cache_key"] = _mk
                st.rerun()
        if not models_list and _models_diag:
            with st.expander("Why model list failed", expanded=True):
                st.code(_models_diag)
            if backend == "openai":
                st.caption(
                    "Tip: use host + port only (no path), e.g. `http://127.0.0.1:8000`. "
                    "For Ollama, switch Backend to **ollama**."
                )
            else:
                st.caption(
                    "Tip: Base URL should be the Ollama listen address (often `http://127.0.0.1:11434`)."
                )
        model_override = st.text_input(
            "Override model id (optional)", value="", key="model_override"
        )
        if models_list:
            model_pick = st.selectbox(
                "Model (from API)", models_list, key="model_from_api"
            )
            model = (model_override.strip() or model_pick).strip()
        else:
            model_manual = st.text_input(
                "Model (manual)",
                value="DeepHat/DeepHat-V1-7B",
                key="model_manual_entry",
            )
            model = (model_override.strip() or model_manual.strip()).strip()

        st.header("Agent")
        base_dir = st.text_input(
            "Workspace (--base-dir)",
            value=".",
            help="Relative paths in tools resolve from this directory. "
            "“.” means the directory you started `streamlit run` from.",
        )
        workspace_resolved = Path(base_dir).expanduser().resolve()
        st.caption(f"Tools use absolute root: `{workspace_resolved}`")
        allow_shell = st.checkbox("Allow shell (`run_terminal_command`)", value=False)
        max_steps = st.number_input(
            "Max inner steps / turn", min_value=1, max_value=64, value=8
        )
        max_tokens = st.number_input(
            "max_tokens / num_predict (0 = omit)", min_value=0, value=1024
        )
        context_limit = st.number_input("Context limit (budget)", min_value=512, value=16384)
        st.caption(
            f"Temperature ({TEMP_MIN}–{TEMP_MAX}, step {TEMP_STEP}); many servers clamp or ignore."
        )
        tcol1, tcol2, tcol3 = st.columns([1, 1, 4])
        with tcol1:
            if st.button("−", key="temp_minus", help=f"Lower (min {TEMP_MIN})"):
                st.session_state.temperature_value = max(
                    TEMP_MIN,
                    round(st.session_state.temperature_value - TEMP_STEP, 2),
                )
        with tcol2:
            if st.button("+", key="temp_plus", help=f"Raise (max {TEMP_MAX})"):
                st.session_state.temperature_value = min(
                    TEMP_MAX,
                    round(st.session_state.temperature_value + TEMP_STEP, 2),
                )
        with tcol3:
            st.markdown(f"**{st.session_state.temperature_value:.2f}**")
        temperature = float(st.session_state.temperature_value)
        tool_list_cap = st.number_input("tool list cap", min_value=5, value=40)
        tool_chars_cap = st.number_input("tool chars cap", min_value=500, value=3500)

        st.header("System prompt")
        system_ta = st.text_area(
            "Extra instructions (merged with built-in agent prompt)",
            value=STREAMLIT_DEFAULT_TOOL_DISCIPLINE,
            height=120,
            help=(
                "Prepended in the chat as: built-in DEFAULT_SYSTEM from offsec_agent_loop, "
                "then this text. If you clear this box, the default anti-hallucination block "
                "is still applied."
            ),
        )

        if st.button("Reset chat", type="primary"):
            _init_session(system_ta, workspace_resolved)
            st.rerun()

    if "agent_messages" not in st.session_state:
        _init_session(system_ta, workspace_resolved)

    saved_ws = st.session_state.get("workspace_root")
    if (
        saved_ws
        and saved_ws != str(workspace_resolved)
        and st.session_state.get("agent_messages")
    ):
        st.sidebar.warning(
            "Workspace path changed since this chat started. Click **Reset chat** "
            "so the system prompt matches the new root."
        )

    for entry in st.session_state.chat_ui:
        role = entry["role"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(entry["content"])
        elif role == "assistant":
            with st.chat_message("assistant"):
                if entry.get("step"):
                    st.caption(f"Inner step {entry['step']}")
                st.markdown(entry.get("content") or "_empty_")
        elif role == "tool":
            with st.chat_message("assistant"):
                with st.expander(f"Tool: `{entry['name']}`"):
                    st.markdown("**Arguments**")
                    st.json(entry.get("args") or {})
                    st.markdown("**Result**")
                    st.json(entry.get("result") or {})
        elif role == "log":
            with st.chat_message("assistant"):
                with st.expander("Engine log", expanded=False):
                    st.code(entry.get("text") or "", language="text")

    user_prompt = st.chat_input("Message the agent…")
    if not user_prompt:
        return

    st.session_state.chat_ui.append({"role": "user", "content": user_prompt})
    st.session_state.agent_messages.append({"role": "user", "content": user_prompt})

    obs = CollectObserver()
    try:
        with st.spinner("Running agent…"):
            rc = run_agent_steps(
                backend=backend,
                url=chat_endpoint(base_url, backend),
                model=model.strip(),
                messages=st.session_state.agent_messages,
                tools=build_tools(allow_shell=allow_shell),
                max_steps=int(max_steps),
                base_dir=workspace_resolved,
                allow_shell=allow_shell,
                api_key=api_key.strip() or None,
                max_tokens=int(max_tokens),
                temperature=temperature,
                context_limit=int(context_limit),
                max_list_entries=int(tool_list_cap),
                max_tool_chars=int(tool_chars_cap),
                observer=obs,
            )
    except Exception as e:  # noqa: BLE001
        st.session_state.chat_ui.pop()
        st.session_state.agent_messages.pop()
        st.error(f"{type(e).__name__}: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return

    for row in obs.rows:
        tag = row[0]
        if tag == "assistant":
            _, step, _max_s, content = row
            st.session_state.chat_ui.append(
                {
                    "role": "assistant",
                    "content": content,
                    "step": step,
                }
            )
        elif tag == "tool":
            _, name, args, result = row
            st.session_state.chat_ui.append(
                {
                    "role": "tool",
                    "name": name,
                    "args": args,
                    "result": result,
                }
            )
        elif tag == "trunc":
            st.session_state.chat_ui.append(
                {
                    "role": "assistant",
                    "content": "_Tool output truncated for LLM context._",
                    "step": None,
                }
            )

    if obs.log_chunks:
        st.session_state.chat_ui.append(
            {"role": "log", "text": "\n".join(obs.log_chunks)}
        )

    if rc != 0:
        st.warning(f"Agent returned exit code {rc}")

    st.rerun()


if __name__ == "__main__":
    main()
