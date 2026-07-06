# Pinned: Cloud Run L4 (driver 535 / CUDA 12.2) fails to detect/use the GPU on
# every Ollama release from v0.30.0 onward (confirmed broken through v0.31.1,
# the latest as of 2026-07-05: "could not determine compute capability" /
# "device kernel image is invalid", falls back to CPU). Root cause: v0.30
# switched to compressed CUDA kernels requiring driver 550+; see
# https://github.com/ollama/ollama/issues/16449 (closed as WAI, driver 535
# will not be supported again in binary releases). v0.24.0 is the most
# recent release confirmed working on L4 for both Qwen3MoE and Gemma 4.
FROM ollama/ollama:0.24.0

ARG MODEL_NAME
ARG ENABLE_TOOLS_TEMPLATE_PATCH=false
ARG PATCHED_MODEL_NAME=
COPY Modelfile.tools.template /tmp/Modelfile.tools.template

RUN ollama serve & sleep 5 && \
    ollama pull "${MODEL_NAME}" && \
    if [ "${ENABLE_TOOLS_TEMPLATE_PATCH}" = "true" ]; then \
      TARGET_MODEL="${PATCHED_MODEL_NAME}"; \
      if [ -z "${TARGET_MODEL}" ]; then TARGET_MODEL="${MODEL_NAME}-tools"; fi; \
      sed "s|__BASE_MODEL__|${MODEL_NAME}|g" /tmp/Modelfile.tools.template > /tmp/Modelfile && \
      ollama create "${TARGET_MODEL}" -f /tmp/Modelfile; \
    fi && \
    pkill ollama

EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama", "serve"]
