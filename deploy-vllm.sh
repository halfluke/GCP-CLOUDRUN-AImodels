#!/usr/bin/env bash
set -euo pipefail

# Deploy vLLM OpenAI-compatible API to Cloud Run (GPU).
# Usage:
#   MODEL_ID="DeepHat/DeepHat-V1-7B" SERVICE="deephat-vllm-7b" ./deploy-vllm.sh
#
# Optional:
#   PROJECT_ID, REGION, REPO, GPU_TYPE, GPU_COUNT, MEMORY, CPU, MIN_INSTANCES,
#   MAX_INSTANCES, TIMEOUT, ALLOW_UNAUTH, HUGGING_FACE_HUB_TOKEN, MAX_MODEL_LEN,
#   DTYPE, GPU_MEMORY_UTILIZATION

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"
REPO="${REPO:-ai-models}"
SERVICE="${SERVICE:-vllm-model}"

MODEL_ID="${MODEL_ID:-DeepHat/DeepHat-V1-7B}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
MEMORY="${MEMORY:-32Gi}"
CPU="${CPU:-8}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-3600}"
ALLOW_UNAUTH="${ALLOW_UNAUTH:-false}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_ENABLE_CUDA_COMPATIBILITY="${VLLM_ENABLE_CUDA_COMPATIBILITY:-0}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib/x86_64-linux-gnu}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

if [[ "${PROJECT_ID}" == "your-project-id" ]]; then
  echo "PROJECT_ID is required. Example:"
  echo "PROJECT_ID=\"my-project\" MODEL_ID=\"DeepHat/DeepHat-V1-7B\" SERVICE=\"deephat-vllm-7b\" ./deploy-vllm.sh"
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
echo

echo "==> Checking required tools"
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

echo "==> Building image in Cloud Build (Dockerfile.vllm)"
TMP_CLOUDBUILD_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CLOUDBUILD_CONFIG}"' EXIT

cat > "${TMP_CLOUDBUILD_CONFIG}" <<'EOF'
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - -f
      - Dockerfile.vllm
      - -t
      - ${_IMAGE_URI}
      - .
images:
  - ${_IMAGE_URI}
EOF

gcloud builds submit . \
  --config "${TMP_CLOUDBUILD_CONFIG}" \
  --region "${REGION}" \
  --substitutions "_IMAGE_URI=${IMAGE_URI}"

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
)

ENV_VARS="MODEL_ID=${MODEL_ID},MAX_MODEL_LEN=${MAX_MODEL_LEN},DTYPE=${DTYPE},GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION},VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY},LD_LIBRARY_PATH=${LD_LIBRARY_PATH},TOOL_CALL_PARSER=${TOOL_CALL_PARSER},ENABLE_AUTO_TOOL_CHOICE=true"
if [[ -n "${HUGGING_FACE_HUB_TOKEN}" ]]; then
  ENV_VARS+=",HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}"
fi
DEPLOY_ARGS+=(--set-env-vars "${ENV_VARS}")

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
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello from Cloud Run vLLM\"}],
    \"max_tokens\": 128
  }"
echo
echo
echo "Tip: tail logs with:"
echo "gcloud beta run services logs tail ${SERVICE} --region ${REGION}"
