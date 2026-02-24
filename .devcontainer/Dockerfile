FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Required for cloning agent-framework from GitHub (it uses Git LFS)
RUN apt-get update && apt-get install -y git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

# Required for building Rust-based Python packages (e.g. base2048 via pyrit/azure-ai-evaluation[redteam])
ENV RUSTUP_HOME="/usr/local/rustup" CARGO_HOME="/usr/local/cargo"
ENV PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock /tmp/uv-tmp/

RUN cd /tmp/uv-tmp && uv sync --no-cache \
    && rm -rf /tmp/uv-tmp
