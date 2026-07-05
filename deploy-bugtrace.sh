#!/usr/bin/env bash
set -euo pipefail

# Deploy BugTrace Apex G4 26B Q4 to Cloud Run (Ollama).
# Uses ollama pull hf.co/ during Cloud Build to bake the model into the image.
#
# Usage:
#   HUGGING_FACE_HUB_TOKEN="hf_..." PROJECT_ID="your-project-id" ./deploy-bugtrace.sh
#
# Optional overrides:
#   SERVICE, OLLAMA_MODEL, GPU_TYPE, MEMORY, MIN_INSTANCES, BUILD_TIMEOUT

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"
REPO="${REPO:-ai-models}"
SERVICE="${SERVICE:-bugtrace-apex-26b}"
OLLAMA_MODEL="${OLLAMA_MODEL:-hf.co/BugTraceAI/BugTraceAI-Apex-G4-26B-Q4:latest}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
MEMORY="${MEMORY:-32Gi}"
CPU="${CPU:-8}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-3600}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-14400s}"
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

if [[ -z "${HUGGING_FACE_HUB_TOKEN}" ]]; then
  echo "HUGGING_FACE_HUB_TOKEN is required."
  echo "Usage: HUGGING_FACE_HUB_TOKEN=hf_... PROJECT_ID=... ./deploy-bugtrace.sh"
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

echo "==> BugTrace Apex G4 26B Q4 (Ollama hf.co pull)"
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "SERVICE=${SERVICE}"
echo "OLLAMA_MODEL=${OLLAMA_MODEL}"
echo "IMAGE_URI=${IMAGE_URI}"
echo "BUILD_TIMEOUT=${BUILD_TIMEOUT}"
echo

command -v gcloud >/dev/null 2>&1 || { echo "gcloud not found"; exit 1; }

gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud config set run/region "${REGION}" >/dev/null

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

if ! gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="LLM images for Cloud Run"
fi

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

TMP_CLOUDBUILD_CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "${TMP_CLOUDBUILD_CONFIG}"' EXIT

cat > "${TMP_CLOUDBUILD_CONFIG}" << YAML
timeout: ${BUILD_TIMEOUT}
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -f
      - Dockerfile.bugtrace
      - --build-arg
      - HF_TOKEN=\${_HF_TOKEN}
      - --build-arg
      - OLLAMA_MODEL=${OLLAMA_MODEL}
      - -t
      - ${IMAGE_URI}
      - .
images:
  - ${IMAGE_URI}
YAML

echo "==> Building image in Cloud Build (~14 GB pull; expect 1-3 hours)"
gcloud builds submit . \
  --config "${TMP_CLOUDBUILD_CONFIG}" \
  --region "${REGION}" \
  --substitutions "_HF_TOKEN=${HUGGING_FACE_HUB_TOKEN}"

echo "==> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --gpu "${GPU_COUNT}" \
  --gpu-type "${GPU_TYPE}" \
  --no-gpu-zonal-redundancy \
  --memory "${MEMORY}" \
  --cpu "${CPU}" \
  --no-cpu-throttling \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  --port 11434 \
  --timeout "${TIMEOUT}" \
  --no-allow-unauthenticated \
  --set-env-vars "OLLAMA_LLM_LIBRARY=cuda_v13"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"

echo
echo "==> Deployment complete"
echo "Service URL: ${SERVICE_URL}"
echo "Model name:  ${OLLAMA_MODEL}"
echo
echo "==> Test (wake from cold start, may take 2-3 min):"
echo "curl -sS \"${SERVICE_URL}/api/generate\" \\"
echo "  -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"${OLLAMA_MODEL}\", \"prompt\": \"Say hi.\", \"stream\": false}'"
echo
echo "==> Benchmark:"
echo "cd ~/Downloads/redteam-ai-benchmark"
echo "git checkout feature/cloudrun-v2"
echo "./scripts/run_bugtrace_baseline.sh"
echo "# or: uv run run_benchmark.py run ollama -m \"${OLLAMA_MODEL}\" -e \"${SERVICE_URL}\" --config configs/cloudrun_ollama_bugtrace.yaml"
