# syntax=docker/dockerfile:1

# ── Build stage ─────────────────────────────────────────────────────
FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY . .
RUN pip install -e ".[all]"

# Default ML packages available to agent-generated code inside the container.
# torch/torchvision/torchaudio are installed as CPU-only wheels to keep the
# image size reasonable — containers have no GPU access by default. Users who
# need CUDA should build a derivative image.
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch torchvision torchaudio \
    && pip install \
        numpy \
        matplotlib \
        einops \
        pandas \
        scikit-learn


# ── Runtime stage ───────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    DATA_PATH=/home/onit/data \
    DOCUMENTS_PATH=/home/onit/documents \
    HOME=/home/onit

# Minimal runtime deps only — no compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 onit \
    && useradd --uid 1000 --gid 1000 --home /home/onit --shell /bin/bash onit \
    && mkdir -p /home/onit/app /home/onit/data /home/onit/documents /home/onit/.onit \
    && chown -R onit:onit /home/onit /opt

COPY --from=build --chown=onit:onit /opt/venv /opt/venv
COPY --from=build --chown=onit:onit /build /home/onit/app

WORKDIR /home/onit/app
USER onit

EXPOSE 9000

ENTRYPOINT ["/usr/bin/tini", "--", "onit"]
