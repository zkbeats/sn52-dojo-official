FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PATH="/root/.cargo/bin/:$PATH"
ENV UV_SYSTEM_PYTHON=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY . .
RUN chmod +x entrypoints.sh

ARG TARGETPLATFORM

RUN echo "Building for TARGETPLATFORM: $TARGETPLATFORM"

# jank because pytorch has different versions for cpu for darwin VS linux, see pyproject.toml for specifics
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    uv pip install --no-cache -e . --find-links https://download.pytorch.org/whl/torch_stable.html; \
    else \
    uv pip install --no-cache -e .; \
    fi

ENTRYPOINT ["./entrypoints.sh"]
