#!/usr/bin/env bash
set -euo pipefail

# Deploy Tongyi DeepResearch GGUF to Cloud Run (Ollama).
# Uses Dockerfile.tongyi: downloads the GGUF in Cloud Build, then ollama create.
#
# Usage:
#   PROJECT_ID=your-project-id ./deploy-tongyi.sh
#
# Optional overrides:
#   SERVICE, OLLAMA_MODEL_NAME, GGUF_URL, MEMORY, MIN_INSTANCES, BUILD_TIMEOUT

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"
REPO="${REPO:-ai-models}"
SERVICE="${SERVICE:-tongyi-deepresearch-iq2s}"
OLLAMA_MODEL_NAME="${OLLAMA_MODEL_NAME:-tongyi-deepresearch-iq2s}"
GGUF_FILE="${GGUF_FILE:-Alibaba-NLP_Tongyi-DeepResearch-30B-A3B-IQ2_S.gguf}"
GGUF_URL="${GGUF_URL:-https://huggingface.co/bartowski/Alibaba-NLP_Tongyi-DeepResearch-30B-A3B-GGUF/resolve/main/${GGUF_FILE}}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
MEMORY="${MEMORY:-32Gi}"
CPU="${CPU:-8}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-3600}"
ALLOW_UNAUTH="${ALLOW_UNAUTH:-false}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-7200s}"
# Cloud Run L4 (driver 535 / CUDA 12.2) needs cuda_v13; cuda_v12 hits "device kernel image is invalid"
SET_ENV_VARS="${SET_ENV_VARS:-OLLAMA_LLM_LIBRARY=cuda_v13}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

echo "==> Tongyi DeepResearch (manual GGUF bake)"
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "SERVICE=${SERVICE}"
echo "OLLAMA_MODEL_NAME=${OLLAMA_MODEL_NAME}"
echo "GGUF_FILE=${GGUF_FILE}"
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
    --description="Ollama model images for Cloud Run"
fi

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

TMP_CLOUDBUILD_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CLOUDBUILD_CONFIG}"' EXIT

cat > "${TMP_CLOUDBUILD_CONFIG}" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -f
      - Dockerfile.tongyi
      - --build-arg
      - GGUF_FILE=${GGUF_FILE}
      - --build-arg
      - GGUF_URL=${GGUF_URL}
      - --build-arg
      - OLLAMA_MODEL_NAME=${OLLAMA_MODEL_NAME}
      - -t
      - ${IMAGE_URI}
      - .
timeout: ${BUILD_TIMEOUT}
images:
  - ${IMAGE_URI}
EOF

echo "==> Building image (downloads ~8.7 GB GGUF in Cloud Build; may take 20-40 min)"
gcloud builds submit . \
  --config "${TMP_CLOUDBUILD_CONFIG}" \
  --region "${REGION}"

echo "==> Deploying Cloud Run service"
DEPLOY_ARGS=(
  run deploy "${SERVICE}"
  --image "${IMAGE_URI}"
  --region "${REGION}"
  --gpu "${GPU_COUNT}"
  --gpu-type "${GPU_TYPE}"
  --memory "${MEMORY}"
  --cpu "${CPU}"
  --no-cpu-throttling
  --min-instances "${MIN_INSTANCES}"
  --max-instances "${MAX_INSTANCES}"
  --port 11434
  --timeout "${TIMEOUT}"
  --no-gpu-zonal-redundancy
)

if [[ "${ALLOW_UNAUTH}" == "true" ]]; then
  DEPLOY_ARGS+=(--allow-unauthenticated)
else
  DEPLOY_ARGS+=(--no-allow-unauthenticated)
fi

if [[ -n "${SET_ENV_VARS}" ]]; then
  DEPLOY_ARGS+=(--set-env-vars "${SET_ENV_VARS}")
fi

gcloud "${DEPLOY_ARGS[@]}"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"

echo
echo "==> Deployment complete"
echo "Service URL: ${SERVICE_URL}"
echo "Ollama model name: ${OLLAMA_MODEL_NAME}"
echo
echo "==> Test call (authenticated)"
curl -sS -X POST "${SERVICE_URL}/api/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -d "{
    \"model\": \"${OLLAMA_MODEL_NAME}\",
    \"prompt\": \"Say hello in one sentence.\",
    \"stream\": false
  }"
echo
echo
echo "Proxy locally:"
echo "  PROJECT_ID=${PROJECT_ID} ./proxy.sh ${SERVICE}"
echo
echo "Benchmark:"
echo "  uv run run_benchmark.py run ollama -m \"${OLLAMA_MODEL_NAME}\" --config configs/cloudrun_ollama.yaml"
echo
echo "Logs:"
echo "  gcloud beta run services logs tail ${SERVICE} --region ${REGION}"
