# Development container for MixedHierarchyGames.jl
# Includes Julia, development tools (git, gh, claude), and pre-compiled dependencies
#
# Security features:
# - Runs as non-root user (devuser)
# - Minimal sudo access (only for package management)
# - No SUID/SGID binaries
# - Secure environment variables
# - Minimal installed packages

# NOTE: Forces linux/amd64 platform because PATHSolver.jl only provides x86_64 binaries.
# On ARM64 hosts (Apple Silicon), Docker will use Rosetta/QEMU emulation.
FROM --platform=linux/amd64 julia:1.11

# Create non-root user (required for claude --dangerously-skip-permissions)
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID devuser \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash devuser

# Set secure environment variables
ENV JULIA_DEPOT_PATH=/home/devuser/.julia
ENV PATH="/home/devuser/.local/bin:${PATH}"
ENV HOME=/home/devuser

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    gnupg \
    ripgrep \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove SUID/SGID bits from all binaries (security hardening)
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory with correct ownership
RUN mkdir -p /workspace && chown devuser:devuser /workspace

# Switch to non-root user for remaining setup
USER devuser
WORKDIR /home/devuser

# Create .claude directory for Claude Code configuration
RUN mkdir -p /home/devuser/.claude

# Create .ssh directory with correct permissions for SSH agent forwarding
RUN mkdir -p /home/devuser/.ssh && chmod 700 /home/devuser/.ssh

# Create .config directory for gh CLI
RUN mkdir -p /home/devuser/.config

# Install Claude Code CLI (native installer) as non-root user
RUN curl -fsSL https://claude.ai/install.sh | bash

# Install beads (bd) CLI for work tracking
USER root
ARG BEADS_VERSION=0.49.4
RUN curl -fsSL "https://github.com/steveyegge/beads/releases/download/v${BEADS_VERSION}/beads_${BEADS_VERSION}_linux_amd64.tar.gz" | tar -xz -C /usr/local/bin bd
USER devuser

# Create working directory
WORKDIR /workspace

# Copy project files with correct ownership (done as root, then chown)
# Note: COPY --chown works with USER directive
COPY --chown=devuser:devuser Project.toml ./
COPY --chown=devuser:devuser test/Project.toml test/Project.toml

# Add General registry and install packages (precompilation happens at runtime)
RUN julia --project=. -e ' \
    using Pkg; \
    Pkg.Registry.add("General"); \
    Pkg.resolve(); \
    Pkg.instantiate(); \
    '

# Copy source files for precompilation
COPY --chown=devuser:devuser src/ src/

# Precompile packages (downloads PATH binaries and LUSOL)
RUN julia --project=. -e ' \
    using Pkg; \
    Pkg.precompile(); \
    '

# Verify PATH solver is available
RUN julia --project=. -e ' \
    using PATHSolver; \
    @info "PATHSolver loaded successfully"; \
    '

# Security: Set restrictive umask
RUN echo "umask 027" >> /home/devuser/.bashrc

# Labels for security scanning and metadata
LABEL org.opencontainers.image.title="MixedHierarchyGames.jl Development Container"
LABEL org.opencontainers.image.description="Julia development environment with Claude Code CLI"
LABEL org.opencontainers.image.vendor="CLeAR Lab"
LABEL security.non-root="true"

# Set default command
CMD ["julia", "--project=."]
