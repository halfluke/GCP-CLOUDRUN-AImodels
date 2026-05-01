#!/usr/bin/env bash
set -euo pipefail

# Deploy vLLM with model files pre-baked into the image.
# Usage:
#   PROJECT_ID="your-project-id" \
#   MODEL_ID="huihui-ai/Qwen2.5-7B-Instruct-abliterated" \
#   HUGGING_FACE_HUB_TOKEN="hf_xxx" \
#   SERVICE="qwen25-7b-abliterated-vllm-prebaked" \
#   ./deploy-vllm-prebaked.sh

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"
REPO="${REPO:-ai-models}"
SERVICE="${SERVICE:-vllm-prebaked-model}"

MODEL_ID="${MODEL_ID:-DeepHat/DeepHat-V1-7B}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
MEMORY="${MEMORY:-32Gi}"
CPU="${CPU:-8}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-3600}"
# Cloud Build default is 3600s; large HF snapshot_download often needs longer.
CLOUD_BUILD_TIMEOUT="${CLOUD_BUILD_TIMEOUT:-10800s}"
ALLOW_UNAUTH="${ALLOW_UNAUTH:-false}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

if [[ "${PROJECT_ID}" == "your-project-id" ]]; then
  echo "PROJECT_ID is required."
  exit 1
fi

if [[ -z "${HUGGING_FACE_HUB_TOKEN}" ]]; then
  echo "HUGGING_FACE_HUB_TOKEN is required for prebaked builds."
  echo "Create a read token at: https://huggingface.co/settings/tokens"
  exit 1
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

echo "==> Using configuration"
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "REPO=${REPO}"
echo "SERVICE=${SERVICE}"
echo "MODEL_ID=${MODEL_ID}"
echo "IMAGE_URI=${IMAGE_URI}"
echo "CLOUD_BUILD_TIMEOUT=${CLOUD_BUILD_TIMEOUT}"
echo

command -v gcloud >/dev/null 2>&1 || { echo "gcloud not found"; exit 1; }

echo "==> Setting active project and region"
gcloud config set project "${PROJECT_ID}" >/dev/null
gcloud config set run/region "${REGION}" >/dev/null

echo "==> Enabling required APIs"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

echo "==> Ensuring Artifact Registry repo exists"
if ! gcloud artifacts repositories describe "${REPO}" --location "${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="LLM images for Cloud Run"
fi

echo "==> Configuring Docker auth helper for Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "==> Building prebaked image in Cloud Build (Dockerfile.vllm.prebaked)"
TMP_CLOUDBUILD_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CLOUDBUILD_CONFIG}"' EXIT

{
  echo "timeout: ${CLOUD_BUILD_TIMEOUT}"
  cat <<'EOF'
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -f
      - Dockerfile.vllm.prebaked
      - --build-arg
      - MODEL_ID=${_MODEL_ID}
      - --build-arg
      - HF_TOKEN=${_HF_TOKEN}
      - -t
      - ${_IMAGE_URI}
      - .
images:
  - ${_IMAGE_URI}
EOF
} > "${TMP_CLOUDBUILD_CONFIG}"

gcloud builds submit . \
  --config "${TMP_CLOUDBUILD_CONFIG}" \
  --region "${REGION}" \
  --substitutions "_IMAGE_URI=${IMAGE_URI},_MODEL_ID=${MODEL_ID},_HF_TOKEN=${HUGGING_FACE_HUB_TOKEN}"

echo "==> Deploying Cloud Run service"
DEPLOY_ARGS=(
  run deploy "${SERVICE}"
  --image "${IMAGE_URI}"
  --region "${REGION}"
  --gpu "${GPU_COUNT}"
  --gpu-type "${GPU_TYPE}"
  --no-gpu-zonal-redundancy
  --memory "${MEMORY}"
  --cpu "${CPU}"
  --no-cpu-throttling
  --min-instances "${MIN_INSTANCES}"
  --max-instances "${MAX_INSTANCES}"
  --port 8080
  --timeout "${TIMEOUT}"
  --set-env-vars "MAX_MODEL_LEN=${MAX_MODEL_LEN},DTYPE=${DTYPE},GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION},VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY},TOOL_CALL_PARSER=${TOOL_CALL_PARSER},ENABLE_AUTO_TOOL_CHOICE=true"
)

if [[ "${ALLOW_UNAUTH}" == "true" ]]; then
  DEPLOY_ARGS+=(--allow-unauthenticated)
else
  DEPLOY_ARGS+=(--no-allow-unauthenticated)
fi

gcloud "${DEPLOY_ARGS[@]}"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"

echo
echo "==> Deployment complete"
echo "Service URL: ${SERVICE_URL}"
echo
echo "==> Test call (authenticated, OpenAI-compatible)"
curl -sS -X POST "${SERVICE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -d "{
    \"model\": \"${MODEL_ID}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello from Cloud Run vLLM (prebaked)\"}],
    \"max_tokens\": 128
  }"
echo
