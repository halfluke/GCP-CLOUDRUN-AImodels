#!/usr/bin/env bash
set -euo pipefail

# Deploy Ollama model container to Google Cloud Run (GPU) using Cloud Build.
# Usage:
#   ./deploy.sh
#   PROJECT_ID=my-project SERVICE=my-ollama MODEL_NAME=deepseek-r1:8b ./deploy.sh
# Optional: SET_ENV_VARS='FOO=bar,BAZ=1' ./deploy.sh

PROJECT_ID="${PROJECT_ID:-your-project-id}"
REGION="${REGION:-europe-west1}"
REPO="${REPO:-ai-models}"
SERVICE="${SERVICE:-ollama-model}"
MODEL_NAME="${MODEL_NAME:-}"
ENABLE_TOOLS_TEMPLATE_PATCH="${ENABLE_TOOLS_TEMPLATE_PATCH:-false}"
PATCHED_MODEL_NAME="${PATCHED_MODEL_NAME:-}"
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
GPU_COUNT="${GPU_COUNT:-1}"
MEMORY="${MEMORY:-32Gi}"
CPU="${CPU:-8}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
TIMEOUT="${TIMEOUT:-3600}"
ALLOW_UNAUTH="${ALLOW_UNAUTH:-false}"
# No OLLAMA_LLM_LIBRARY override: the image is pinned to ollama/ollama:0.24.0,
# which auto-detects cuda_v12 correctly on Cloud Run L4. Newer Ollama releases
# (v0.30.0+) broke GPU detection on L4 outright, so this is no longer a
# backend-selection workaround, it's a version pin (see Dockerfile).
SET_ENV_VARS="${SET_ENV_VARS:-}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "MODEL_NAME is required."
  echo "Example:"
  echo "MODEL_NAME=\"hf.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-Uncensored-GGUF:Q6_K\" SERVICE=\"deepseek-r1-14b-q6k\" ./deploy.sh"
  exit 1
fi

echo "==> Using configuration"
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "REPO=${REPO}"
echo "SERVICE=${SERVICE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "ENABLE_TOOLS_TEMPLATE_PATCH=${ENABLE_TOOLS_TEMPLATE_PATCH}"
echo "PATCHED_MODEL_NAME=${PATCHED_MODEL_NAME:-<auto>}"
echo "ALLOW_UNAUTH=${ALLOW_UNAUTH}"
echo "SET_ENV_VARS=${SET_ENV_VARS:-<none>}"
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
    --description="Ollama model images for Cloud Run"
fi

echo "==> Configuring Docker auth helper for Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

echo "==> Building image in Cloud Build (remote build with MODEL_NAME build arg)"
TMP_CLOUDBUILD_CONFIG="$(mktemp)"
trap 'rm -f "${TMP_CLOUDBUILD_CONFIG}"' EXIT

cat > "${TMP_CLOUDBUILD_CONFIG}" <<'EOF'
steps:
  - name: gcr.io/cloud-builders/docker
    args:
      - build
      - --build-arg
      - MODEL_NAME=${_MODEL_NAME}
      - --build-arg
      - ENABLE_TOOLS_TEMPLATE_PATCH=${_ENABLE_TOOLS_TEMPLATE_PATCH}
      - --build-arg
      - PATCHED_MODEL_NAME=${_PATCHED_MODEL_NAME}
      - -t
      - ${_IMAGE_URI}
      - .
images:
  - ${_IMAGE_URI}
EOF

gcloud builds submit . \
  --config "${TMP_CLOUDBUILD_CONFIG}" \
  --region "${REGION}" \
  --substitutions "_MODEL_NAME=${MODEL_NAME},_ENABLE_TOOLS_TEMPLATE_PATCH=${ENABLE_TOOLS_TEMPLATE_PATCH},_PATCHED_MODEL_NAME=${PATCHED_MODEL_NAME},_IMAGE_URI=${IMAGE_URI}"

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
else
  # Cloud Run persists env vars across deploys unless explicitly cleared.
  # Without this, a stale OLLAMA_LLM_LIBRARY from a previous revision would
  # silently survive and override the image's correct auto-detected backend.
  DEPLOY_ARGS+=(--clear-env-vars)
fi

gcloud "${DEPLOY_ARGS[@]}"

SERVICE_URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')"

echo
echo "==> Deployment complete"
echo "Service URL: ${SERVICE_URL}"
echo
echo "==> Test call (authenticated)"
curl -sS -X POST "${SERVICE_URL}/api/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -d "{
    \"model\": \"${MODEL_NAME}\",
    \"prompt\": \"Hello from Cloud Run\",
    \"stream\": false
  }"
echo
echo
echo "Tip: tail logs with:"
echo "gcloud beta run services logs tail ${SERVICE} --region ${REGION}"
