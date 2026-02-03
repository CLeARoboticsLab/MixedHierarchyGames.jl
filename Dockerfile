# Development container for MixedHierarchyGames.jl
# Includes Julia, development tools (git, gh, claude), and pre-compiled dependencies

FROM julia:1.11

# Set environment variables
ENV JULIA_DEPOT_PATH=/opt/julia-depot
ENV PATH_LICENSE_STRING="1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"
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

# Install Claude Code CLI (native installer)
RUN curl -fsSL https://claude.ai/install.sh | bash

# Create working directory
WORKDIR /workspace

# Copy project files for dependency installation
COPY Project.toml ./
COPY test/Project.toml test/Project.toml

# Add General registry and install packages (precompilation happens at runtime)
RUN julia --project=. -e ' \
    using Pkg; \
    Pkg.Registry.add("General"); \
    Pkg.resolve(); \
    Pkg.instantiate(); \
    '

# Set default command
CMD ["julia", "--project=."]
