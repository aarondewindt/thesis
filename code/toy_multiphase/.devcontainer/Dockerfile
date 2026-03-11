FROM rocm/tensorflow:rocm7.2-py3.10-tf2.20-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    git \
    ca-certificates \
    curl \
    build-essential \
    zsh \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && groupadd -g 992 render \
    && groupadd -f video \
    && usermod -aG render,video $USERNAME || true


ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH --system-site-packages \
    && $VENV_PATH/bin/pip install --upgrade pip setuptools wheel \
    && chown -R vscode:vscode ${VENV_PATH}
    
ENV PATH="$VENV_PATH/bin:$PATH"

USER $USERNAME

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
    -p git -x

WORKDIR /workspaces/thesis/code/toy_multiphase
