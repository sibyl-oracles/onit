# syntax=docker/dockerfile:1

# ── Build stage ─────────────────────────────────────────────────────
# Use the CUDA devel base so nvcc + headers are available for native builds.
# Matches the CUDA 12.6 torch wheels pulled below.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Switch apt sources to HTTPS (some networks block outbound port 80 to
# Canonical mirrors). ca-certificates isn't preinstalled, so disable TLS
# verification for this one fetch — it's removed right after.
RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources \
    && echo 'Acquire::https::Verify-Peer "false";' > /etc/apt/apt.conf.d/99no-verify \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        gcc \
        python3-dev \
    && rm -f /etc/apt/apt.conf.d/99no-verify \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY . .
# Non-editable install so the package resolves from site-packages in the
# runtime stage (an editable install pins sys.path to /build, which is
# discarded by the multi-stage copy).
RUN pip install ".[all]"

# Lightweight scientific stack available to agent-generated code by default.
# Small, universally useful, and keeps the image lean. Heavy ML packages
# (torch, transformers, etc) are NOT baked in — the agent installs them on
# demand via `onit-install-ml` into the persistent data volume (PIP_TARGET).
RUN pip install \
        numpy \
        scipy \
        pandas \
        matplotlib \
        scikit-learn


# ── Runtime stage ───────────────────────────────────────────────────
# devel (not runtime) so nvcc + CUDA headers are available to agent-generated
# code that needs to compile CUDA kernels. Bigger image, but user requested
# the full toolkit.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    DATA_PATH=/home/onit/data \
    DOCUMENTS_PATH=/home/onit/documents \
    HOME=/home/onit \
    # Signal to MCP servers that they're running inside the onit container, so
    # the DATA_PATH/DOCUMENTS_PATH allowlist is relaxed — the container itself
    # is the filesystem boundary, and extra host mounts (--container-mount)
    # should be fully accessible to tools.
    ONIT_CONTAINER=1 \
    # The venv at /opt/venv is on the read-only rootfs, so `pip install X` can't
    # write there. Redirect pip's install target and cache to the persistent
    # data volume (mounted writable at runtime). PYTHONPATH makes the extra
    # packages importable by the venv's python.
    PIP_TARGET=/home/onit/data/.python-packages \
    PIP_CACHE_DIR=/home/onit/data/.pip-cache \
    PYTHONPATH=/home/onit/data/.python-packages \
    # Park the Hugging Face cache on the persistent data volume — default is
    # ~/.cache/huggingface which is a 2g tmpfs; model weights would fill it
    # instantly. hf_transfer is installed in the build stage for faster pulls.
    HF_HOME=/home/onit/data/.huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # Redirect XDG config/cache and matplotlib config to the persistent data
    # volume. XDG_CACHE_HOME in particular needs to land on the volume because
    # torch hub / torchvision weights are big (GBs) and the old tmpfs default
    # would fill up. Volume-backed means cache survives container restarts.
    XDG_CONFIG_HOME=/home/onit/data/.config \
    XDG_CACHE_HOME=/home/onit/data/.cache \
    MPLCONFIGDIR=/home/onit/data/.matplotlib

# Minimal runtime deps only — no compiler. See build stage for TLS note.
# ubuntu:24.04 ships a default `ubuntu` user at UID/GID 1000 — remove it
# before creating our own.
RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources \
    && echo 'Acquire::https::Verify-Peer "false";' > /etc/apt/apt.conf.d/99no-verify \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        ca-certificates \
        tini \
        espeak-ng \
        libespeak-ng1 \
        git \
        openssh-client \
        sudo \
        build-essential \
        curl \
        wget \
    && rm -f /etc/apt/apt.conf.d/99no-verify \
    && rm -rf /var/lib/apt/lists/* \
    && (userdel -r ubuntu 2>/dev/null || true) \
    && groupadd --gid 1000 onit \
    && useradd --uid 1000 --gid 1000 --home /home/onit --shell /bin/bash onit \
    # Passwordless sudo for the in-container user. The agent uses this to
    # `apt install` extra system packages at runtime. Note: when the launcher
    # runs the container as the host user's UID/GID, that user isn't `onit`,
    # so sudoers is widened via %sudo and the user is added by the launcher's
    # runtime bootstrap if needed. The NOPASSWD rule below matches by uid 1000
    # which is the default non-bind-mount path.
    && echo 'onit ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/99-onit \
    && echo 'ALL ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/99-any \
    && chmod 0440 /etc/sudoers.d/99-onit /etc/sudoers.d/99-any \
    && mkdir -p /home/onit/app /home/onit/data /home/onit/documents /home/onit/.onit \
    # World-writable on the directories the container process writes to. This
    # lets the launcher run the container as the host user's UID/GID (to match
    # bind-mount ownership) without losing write access to the named data
    # volume, which inherits perms from this image's mount points on first use.
    && chmod 0777 /home/onit /home/onit/data /home/onit/documents /home/onit/.onit \
    # Install a git credential helper that uses $GITHUB_TOKEN (bridged in from
    # the host keychain by container_launcher) instead of prompting at the TTY.
    # Falls through silently when the token is absent, so public clones still
    # work. Scripted on disk (not inline in ~/.gitconfig) because git's config
    # parser strips unescaped double-quotes from value strings, mangling any
    # embedded shell quoting.
    && printf '%s\n' \
        '#!/bin/sh' \
        '[ "$1" = "get" ] || exit 0' \
        '[ -n "$GITHUB_TOKEN" ] || exit 0' \
        'printf "username=x-access-token\npassword=%s\n" "$GITHUB_TOKEN"' \
        > /usr/local/bin/git-credential-onit-github \
    && chmod 0755 /usr/local/bin/git-credential-onit-github \
    && printf '%s\n' \
        '[credential "https://github.com"]' \
        '    helper = onit-github' \
        > /home/onit/.gitconfig \
    && chown -R onit:onit /home/onit /opt

COPY --from=build --chown=onit:onit /opt/venv /opt/venv
COPY --from=build --chown=onit:onit /build /home/onit/app

# Helper the agent invokes to install heavy ML packages (torch, transformers,
# etc) on demand. Goes to the persistent data volume via PIP_TARGET, so the
# install cost is paid once per host. Inlined via BuildKit heredoc so the
# script survives gitignore/dockerignore filters.
COPY <<'EOF' /usr/local/bin/onit-install-ml
#!/bin/bash
# Install heavy ML packages into the persistent data volume on demand.
# Usage: onit-install-ml [preset ...]
#   torch    torch / torchvision / torchaudio (auto-picks wheel matching the
#            host NVIDIA driver's max CUDA, or CPU if no GPU is attached)
#   hf       transformers / datasets / accelerate / tokenizers / safetensors / hf_transfer
#   extras   einops / phonemizer
#   all      torch + hf + extras (default)
set -euo pipefail
presets=("${@:-all}")

have_cuda() {
    [ -e /proc/driver/nvidia/version ] || compgen -G "/dev/nvidia*" >/dev/null 2>&1
}

# Map the host driver's max CUDA version (reported by nvidia-smi) to the
# closest-matching torch wheel index. Newer driver == higher CUDA supported.
# Picking a wheel built for a CUDA version <= driver avoids the "driver too
# old" runtime error from torch.
pick_torch_index() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "cu126"; return
    fi
    local line major minor
    line=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | head -1 || true)
    if [ -z "$line" ]; then
        echo "cu126"; return
    fi
    major=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+' | cut -d. -f1)
    minor=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+' | cut -d. -f2)
    # Available torch wheel indices (as of 2025): cu118, cu121, cu124, cu126, cu128.
    # Pick the highest one the driver can support.
    if [ "$major" -gt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; }; then
        echo "cu128"
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 6 ]; then
        echo "cu126"
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 4 ]; then
        echo "cu124"
    elif [ "$major" -eq 12 ]; then
        echo "cu121"
    else
        echo "cu118"
    fi
}

install_torch() {
    local idx
    if have_cuda; then
        idx=$(pick_torch_index)
        echo ">>> Installing torch ($idx wheels, matched to host driver)"
    else
        idx="cpu"
        echo ">>> Installing torch (CPU wheels — no GPU attached)"
    fi
    # Wipe any prior torch install + bundled nvidia libs from PIP_TARGET so
    # a previous (possibly wrong-CUDA) wheel doesn't leave stale deps next to
    # the new one. `pip uninstall` doesn't work cleanly with --target, so we
    # remove the directories directly; the glob is narrow enough not to touch
    # unrelated packages.
    if [ -n "${PIP_TARGET:-}" ] && [ -d "$PIP_TARGET" ]; then
        rm -rf "$PIP_TARGET"/torch "$PIP_TARGET"/torch-*.dist-info \
               "$PIP_TARGET"/torchvision "$PIP_TARGET"/torchvision-*.dist-info \
               "$PIP_TARGET"/torchaudio "$PIP_TARGET"/torchaudio-*.dist-info \
               "$PIP_TARGET"/nvidia "$PIP_TARGET"/nvidia_*-*.dist-info \
               2>/dev/null || true
    fi
    # IMPORTANT: --index-url only, no --extra-index-url. The torch index
    # mirrors common deps (numpy, pillow, fsspec, jinja2, …); adding pypi
    # as a fallback lets pip pick up a pypi-hosted torch with a newer CUDA
    # runtime (PEP 440 ranks the non-local "2.x" above the "2.x+cu126"
    # local version) which breaks with "NVIDIA driver too old".
    pip install --index-url "https://download.pytorch.org/whl/$idx" \
                torch torchvision torchaudio
}

install_hf() {
    echo ">>> Installing Hugging Face stack"
    pip install transformers datasets accelerate \
                safetensors tokenizers hf_transfer
}

install_extras() {
    echo ">>> Installing einops, phonemizer"
    pip install einops phonemizer
}

for p in "${presets[@]}"; do
    case "$p" in
        torch)  install_torch ;;
        hf)     install_hf ;;
        extras) install_extras ;;
        all)    install_torch; install_hf; install_extras ;;
        *)      echo "Unknown preset: $p. Valid: torch, hf, extras, all" >&2; exit 1 ;;
    esac
done
echo
echo ">>> Done. Installed into $PIP_TARGET (persists across container runs)."
EOF
RUN chmod 0755 /usr/local/bin/onit-install-ml

WORKDIR /home/onit/app
USER onit

EXPOSE 9000

ENTRYPOINT ["/usr/bin/tini", "--", "onit"]
