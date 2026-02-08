# Development container for MixedHierarchyGames.jl
# Includes Julia, development tools (git, gh, claude), and pre-compiled dependencies
#
# Security features:
# - Runs as non-root user (devuser)
# - Minimal sudo access (only for SSH socket fix)
# - No SUID/SGID binaries
# - Secure environment variables
# - Minimal installed packages

# NOTE: Forces linux/amd64 platform because PATHSolver.jl only provides x86_64 binaries.
# On ARM64 hosts (Apple Silicon), Docker will use Rosetta/QEMU emulation.
FROM --platform=linux/amd64 julia:1.11

# Create non-root user (required for claude --allow-dangerously-skip-permissions)
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID devuser \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash devuser

# Set secure environment variables
ENV JULIA_DEPOT_PATH=/home/devuser/.julia
ENV PATH="/home/devuser/.local/bin:${PATH}"
ENV HOME=/home/devuser

# Install system dependencies, GitHub CLI, Node.js, and sudo (single apt layer)
# Node.js is required for GSD workflow tools (.claude/get-shit-done/bin/gsd-tools.js)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    gnupg \
    ripgrep \
    openssh-client \
    sudo \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y gh nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove SUID/SGID bits from all binaries (security hardening)
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# Allow devuser to fix SSH socket permissions (Docker Desktop for Mac mounts it as root)
RUN echo "devuser ALL=(root) NOPASSWD: /usr/bin/chmod 666 /run/host-services/ssh-auth.sock" > /etc/sudoers.d/ssh-socket \
    && chmod 440 /etc/sudoers.d/ssh-socket

# Create workspace and user directories with correct ownership (as root)
RUN mkdir -p /workspace && chown devuser:devuser /workspace \
    && mkdir -p /home/devuser/.claude /home/devuser/.cache/claude \
       /home/devuser/.config /home/devuser/.ssh \
    && chmod 700 /home/devuser/.ssh \
    && chown -R devuser:devuser /home/devuser

# Switch to non-root user for remaining setup
USER devuser
WORKDIR /home/devuser

# Install Claude Code CLI (native installer) as non-root user
RUN curl -fsSL https://claude.ai/install.sh | bash

# Install beads (bd) CLI for work tracking
USER root
ARG BEADS_VERSION=0.49.0
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

# Entrypoint: fix SSH socket permissions then exec the command
# Docker Desktop for Mac mounts the SSH agent socket as root:root
COPY --chown=devuser:devuser <<'EOF' /home/devuser/entrypoint.sh
#!/bin/bash
if [ -S "/run/host-services/ssh-auth.sock" ]; then
    sudo chmod 666 /run/host-services/ssh-auth.sock 2>/dev/null || true
fi
exec "$@"
EOF
RUN chmod +x /home/devuser/entrypoint.sh
ENTRYPOINT ["/home/devuser/entrypoint.sh"]

# Labels for security scanning and metadata
LABEL org.opencontainers.image.title="MixedHierarchyGames.jl Development Container"
LABEL org.opencontainers.image.description="Julia development environment with Claude Code CLI"
LABEL org.opencontainers.image.vendor="CLeAR Lab"
LABEL security.non-root="true"

# Set default command
CMD ["julia", "--project=."]
