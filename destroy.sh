#!/usr/bin/env bash
set -euo pipefail

# Delete Cloud Run service for Ollama deployment.
# Usage:
#   ./destroy.sh
#   PROJECT_ID=my-project SERVICE=my-ollama REGION=europe-west1 ./destroy.sh

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-ollama-model}"

if [[ "${PROJECT_ID}" == "your-project-id" ]]; then
  echo "PROJECT_ID is not set. Example:"
  echo "  PROJECT_ID=\"your-project-id\" REGION=\"europe-west1\" SERVICE=\"qwen3-8b\" ./destroy.sh"
  exit 1
fi

echo "==> Using configuration"
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "SERVICE=${SERVICE}"
echo

echo "==> Checking required tools"
command -v gcloud >/dev/null 2>&1 || { echo "gcloud not found"; exit 1; }

echo "==> Setting active project"
gcloud config set project "${PROJECT_ID}" >/dev/null

echo "==> Deleting Cloud Run service (if it exists)"
if gcloud run services describe "${SERVICE}" --region "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud run services delete "${SERVICE}" --region "${REGION}" --project "${PROJECT_ID}" --quiet
  echo "Service deleted."
else
  echo "Service not found, nothing to delete."
fi
