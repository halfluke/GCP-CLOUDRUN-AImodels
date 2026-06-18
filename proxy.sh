#!/usr/bin/env bash
set -euo pipefail

# Helper to proxy Cloud Run inference services to localhost.
# Known shortcuts match README service names; anything else is treated as a raw Cloud Run service id.
# Usage:
#   ./proxy.sh qwen
#   ./proxy.sh deepseek
#   ./proxy.sh redteam    # same as nu11
#   ./proxy.sh deephat    # vLLM — local port defaults to 8080
#   ./proxy.sh <cloud-run-service-name>
#   ./proxy.sh list
# Optional env overrides:
#   PROJECT_ID, REGION, PORT (if unset, port defaults: 11434 for Ollama, 8080 for deephat)

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"

TARGET="${1:-}"

if [[ -z "${TARGET}" ]]; then
  echo "Usage: $0 {qwen|deepseek|redteam|nu11|tongyi|deephat|<cloud-run-service>|list}"
  exit 1
fi

if [[ "${PROJECT_ID}" == "your-project-id" ]]; then
  echo "PROJECT_ID is not set."
  echo "Run with: PROJECT_ID=\"<your-gcp-project-id>\" ./proxy.sh ${TARGET}"
  echo "Example: PROJECT_ID=\"your-gcp-project-id\" ./proxy.sh qwen"
  exit 1
fi

DEFAULT_PORT="11434"

case "${TARGET}" in
  qwen)
    SERVICE="qwen3-8b"
    MODEL_HINT="Ollama Continue apiBase http://127.0.0.1:<port> — model qwen3:8b (no /v1)"
    ;;
  deepseek)
    SERVICE="deepseek-r1-8b"
    MODEL_HINT="Ollama Continue apiBase http://127.0.0.1:<port> — model deepseek-r1:8b"
    ;;
  redteam | nu11)
    SERVICE="nu11-redteamlite-ollama"
    MODEL_HINT="Ollama Continue apiBase http://127.0.0.1:<port> — model f0rc3ps/nu11secur1tyAIRedTeamLite"
    ;;
  tongyi)
    SERVICE="tongyi-deepresearch-iq2s"
    MODEL_HINT="Ollama apiBase http://127.0.0.1:<port> — model tongyi-deepresearch-iq2s (manual GGUF bake)"
    ;;
  deephat)
    SERVICE="deephat-vllm-7b-prebaked"
    MODEL_HINT="OpenAI compat apiBase http://127.0.0.1:<port>/v1 — model DeepHat/DeepHat-V1-7B"
    DEFAULT_PORT="8080"
    ;;
  list)
    gcloud run services list --region "${REGION}" --project "${PROJECT_ID}"
    exit 0
    ;;
  *)
    SERVICE="${TARGET}"
    MODEL_HINT="<set model + apiBase to match this Cloud Run service>"
    ;;
esac

PROXY_PORT="${PORT:-${DEFAULT_PORT}}"

echo "Proxy target: ${TARGET}"
echo "Cloud Run service: ${SERVICE}"
echo "Hint: ${MODEL_HINT//<port>/${PROXY_PORT}}"
echo "Local bind: http://127.0.0.1:${PROXY_PORT}"
echo
echo "Starting proxy (Ctrl+C to stop)..."

gcloud run services proxy "${SERVICE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --port "${PROXY_PORT}"
