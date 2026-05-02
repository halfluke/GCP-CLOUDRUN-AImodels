#!/usr/bin/env bash
set -euo pipefail

# Quick vLLM model smoke test:
#  1) chat-only response
#  2) tool-calling shape (checks message.tool_calls)
#
# Usage examples:
#   MODEL_ID="DeepHat/DeepHat-V1-7B" ./test-vllm-model.sh
#   MODEL_ID="..." VLLM_BASE_URL="http://127.0.0.1:8080" TOOL_CHOICE=required ./test-vllm-model.sh
#   MODEL_ID="..." VLLM_API_KEY="dummy" ./test-vllm-model.sh

MODEL_ID="${MODEL_ID:-}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8080}"
VLLM_API_KEY="${VLLM_API_KEY:-}"
MAX_TOKENS="${MAX_TOKENS:-128}"
# Try TOOL_CHOICE=required if auto leaves tool_calls empty (model + parser dependent).
TOOL_CHOICE="${TOOL_CHOICE:-auto}"

if [[ -z "${MODEL_ID}" ]]; then
  echo "MODEL_ID is required."
  echo "Example: MODEL_ID=\"DeepHat/DeepHat-V1-7B\" ./test-vllm-model.sh"
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required."
  exit 1
fi

AUTH_HEADER=()
if [[ -n "${VLLM_API_KEY}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

CHAT_PAYLOAD="$(cat <<EOF
{
  "model":"${MODEL_ID}",
  "messages":[{"role":"user","content":"Reply with exactly: OK"}],
  "max_tokens":${MAX_TOKENS},
  "stream":false
}
EOF
)"

TOOL_PAYLOAD="$(cat <<EOF
{
  "model":"${MODEL_ID}",
  "messages":[{"role":"user","content":"Use test tool with city=London. Do not answer directly."}],
  "tools":[{"type":"function","function":{"name":"test","description":"test","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
  "tool_choice":"${TOOL_CHOICE}",
  "max_tokens":${MAX_TOKENS},
  "stream":false
}
EOF
)"

echo "==> Chat-only test"
CHAT_RESP="$(curl -sS "${VLLM_BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${CHAT_PAYLOAD}")"
echo "${CHAT_RESP}"
echo

echo "==> Tool-call test (tool_choice=${TOOL_CHOICE})"
TOOL_RESP="$(curl -sS "${VLLM_BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  "${AUTH_HEADER[@]}" \
  -d "${TOOL_PAYLOAD}")"
echo "${TOOL_RESP}"
echo

if command -v jq >/dev/null 2>&1; then
  TOOL_CALLS_LEN="$(echo "${TOOL_RESP}" | jq -r '.choices[0].message.tool_calls | length // 0' 2>/dev/null || echo "0")"
  CONTENT_PREVIEW="$(echo "${TOOL_RESP}" | jq -r '.choices[0].message.content // ""' 2>/dev/null || true)"
  PARSEABLE_TOOL_JSON="$(echo "${TOOL_RESP}" | jq -r '
    .choices[0].message.content as $c
    | if ($c | type) != "string" then
        "false"
      elif ($c | test("<tool_call>"; "i")) and ($c | test("\"name\"")) then
        "true"
      else
        ($c
          | gsub("^```json\\s*"; "")
          | gsub("^```\\s*"; "")
          | gsub("\\s*```$"; "")
        ) as $clean
        | (try ($clean | fromjson) catch null) as $parsed
        | if ($parsed | type) == "array"
             and ($parsed | length) > 0
             and (($parsed[0].name? // null) != null)
             and (($parsed[0].arguments? // null) != null)
          then "true"
          elif (($parsed | type) == "object")
             and (($parsed.name? // null) != null)
             and (($parsed.arguments? // null) != null)
          then "true"
          else "false"
          end
      end
  ' 2>/dev/null || echo "false")"
  echo "==> Parsed summary"
  echo "tool_calls length: ${TOOL_CALLS_LEN}"
  if [[ "${TOOL_CALLS_LEN}" -gt 0 ]]; then
    echo "result: native structured tool_calls detected (Continue-ready)"
  else
    echo "result: no structured tool_calls"
    if [[ "${PARSEABLE_TOOL_JSON}" == "true" ]]; then
      echo "note: parseable tool JSON in message.content (clients need native tool_calls)"
    elif [[ -n "${CONTENT_PREVIEW}" ]]; then
      echo "note: message.content is plain text/non-parseable as tool JSON"
    fi
  fi
else
  echo "jq not found; raw responses shown above."
fi
