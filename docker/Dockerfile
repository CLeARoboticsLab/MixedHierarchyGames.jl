FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg2 lsb-release software-properties-common \
    build-essential git wget locales tzdata sudo \
    python3 python3-pip python3-venv python3-dev \
    pkg-config libssl-dev libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Install ROS 2 Jazzy
RUN curl -sSL http://repo.ros2.org/repos.key | apt-key add - || true && \
    apt-add-repository "deb [arch=amd64] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep python3-colcon-common-extensions \
    ros-jazzy-desktop && \
    rm -rf /var/lib/apt/lists/* && \
    rosdep init || true

# Install Julia 1.10.4
ENV JULIA_VERSION=1.10.4
RUN curl --retry 5 --retry-delay 10 --connect-timeout 30 --max-time 300 \
    -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    -o /tmp/julia.tar.gz && \
    tar -xzf /tmp/julia.tar.gz -C /opt && \
    ln -sf /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia && \
    rm /tmp/julia.tar.gz

# Create developer user
ARG USER=developer
ARG UID=1011
ARG GID=1011
RUN if [ "${UID}" != "0" ] && [ "${GID}" != "0" ]; then \
        groupadd -g ${GID} ${USER} || true; \
        useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER} || true; \
        echo "${USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER} && \
        chmod 0440 /etc/sudoers.d/${USER}; \
    fi

WORKDIR /home/${USER}/workspace
USER ${USER}

# Copy workspace
COPY --chown=${USER}:${USER} . /home/${USER}/workspace

# Create Python venv with system-site-packages (to access ROS packages)
USER root
RUN python3 -m venv --system-site-packages /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/python -m pip install --no-cache-dir numpy julia && \
    (test -f /home/${USER}/workspace/requirements.txt && \
     /opt/venv/bin/python -m pip install --no-cache-dir -r /home/${USER}/workspace/requirements.txt || true) && \
    chown -R ${UID}:${GID} /opt/venv

# Set up Julia with PyCall (needed by PyJulia)
ENV JULIA_SSL_LIBRARY=system
ENV PYTHON_PATH=/opt/venv/bin/python
ENV PATH="/opt/venv/bin:${PATH}"

# Install PyCall in default Julia environment (required by PyJulia)
RUN JULIA_SSL_LIBRARY=system julia --startup-file=no -e "import Pkg; ENV[\"PYTHON\"]=\"${PYTHON_PATH}\"; Pkg.add(\"PyCall\"); Pkg.build(\"PyCall\")" || true

# Install Julia packages in project
RUN cd /home/${USER}/workspace && \
    rm -rf ~/.julia/artifacts/* || true && \
    JULIA_SSL_LIBRARY=system julia --startup-file=no -e "import Pkg; Pkg.activate(\".\"); ENV[\"PYTHON\"]=\"${PYTHON_PATH}\"; Pkg.add(\"PyCall\"); Pkg.build(\"PyCall\"); Pkg.instantiate(); Pkg.precompile()" || true

USER ${USER}

ENV ROS_DOMAIN_ID=0
ENV ROS_NAMESPACE=

# Setup entrypoint
COPY --chown=${USER}:${USER} docker/entrypoint.sh /home/${USER}/entrypoint.sh
RUN chmod +x /home/${USER}/entrypoint.sh

ENTRYPOINT ["/home/developer/entrypoint.sh"]
