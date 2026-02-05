# Development container for MixedHierarchyGames.jl
# Includes Julia, development tools (git, gh, claude), and pre-compiled dependencies
#
# NOTE: Forces linux/amd64 platform because PATHSolver.jl only provides x86_64 binaries.
# On ARM64 hosts (Apple Silicon), Docker will use Rosetta/QEMU emulation.

FROM --platform=linux/amd64 julia:1.11

# Set environment variables
ENV JULIA_DEPOT_PATH=/opt/julia-depot
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    gnupg \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI and make accessible to all users
RUN curl -fsSL https://claude.ai/install.sh | bash && \
    cp /root/.local/bin/claude /usr/local/bin/claude && \
    chmod 755 /usr/local/bin/claude

# Create working directory
WORKDIR /workspace

# Copy project files for dependency installation
COPY Project.toml ./
COPY test/Project.toml test/Project.toml

# Add General registry and install dependencies (without precompiling MixedHierarchyGames)
RUN julia --project=. -e ' \
    using Pkg; \
    Pkg.Registry.add("General"); \
    Pkg.resolve(); \
    Pkg.instantiate(); \
    '

# Copy source files for precompilation
COPY src/ src/

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

# Set default command
CMD ["julia", "--project=."]
