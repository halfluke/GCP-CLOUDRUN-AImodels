#!/usr/bin/env bash
set -euo pipefail

# Restart a Cloud Run Ollama service revision to recover from stuck generations.
# Usage:
#   PROJECT_ID="your-project-id" ./unstick.sh qwen
#   PROJECT_ID="your-project-id" ./unstick.sh deepseek
#   PROJECT_ID="your-project-id" ./unstick.sh redteam
#   PROJECT_ID="your-project-id" ./unstick.sh <service-name>
#
# Optional env:
#   REGION (default: europe-west1)

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"
TARGET="${1:-}"

if [[ -z "${TARGET}" ]]; then
  echo "Usage: $0 {qwen|deepseek|redteam|nu11|deephat|<service-name>}"
  exit 1
fi

if [[ "${PROJECT_ID}" == "your-project-id" ]]; then
  echo "PROJECT_ID is not set."
  echo "Run with: PROJECT_ID=\"<your-gcp-project-id>\" ./unstick.sh ${TARGET}"
  exit 1
fi

case "${TARGET}" in
  qwen) SERVICE="qwen3-8b" ;;
  deepseek) SERVICE="deepseek-r1-8b" ;;
  redteam | nu11) SERVICE="nu11-redteamlite-ollama" ;;
  deephat) SERVICE="deephat-vllm-7b-prebaked" ;;
  *) SERVICE="${TARGET}" ;;
esac

echo "Restarting Cloud Run service revision..."
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "SERVICE=${SERVICE}"
echo

# Triggers a new revision by updating a harmless env var nonce.
# This avoids "No configuration change requested" errors.
gcloud run services update "${SERVICE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --update-env-vars "UNSTICK_NONCE=$(date +%s)" \
  --quiet >/dev/null

SERVICE_URL="$(gcloud run services describe "${SERVICE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format='value(status.url)')"

echo "Service updated. URL: ${SERVICE_URL}"
echo "Tip: re-run proxy after this:"
echo "  PROJECT_ID=\"${PROJECT_ID}\" ./proxy.sh ${TARGET}"
