FROM ollama/ollama:latest

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
