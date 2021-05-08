# Development setup for the trajectory optimization 1 project.
# Includes:
#  - Preconfigured Jupyterlab with my preffered configuration and extentions
#    installed.
#  - Dependencies for GPU access installed, extra configurations are needed
#    when starting a new container, see 'docker-compose.dev.yml'
#  - Dependencies for X forwarding (GUI) to the host, extra configurations
#    are needed when starting a new container, see 'docker-compose.dev.yml'
#  - The project and it's dependencies installed and ready to be used.
#
# Notes:
#  - GUI forwarding only works on Linux hosts.
#  - GPU access only works for NVIDIA GPUs.
#  - GUI and GPU features are mostly based on this article, but some of
#    docker recently changed how containers are given access to GPU's, so
#    it's a bit outdated. I don't remember how I solved it though.
#    https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1


FROM jupyter/tensorflow-notebook

# Define project environment variables
ENV PROJECT_NAME=trajectory_optimization_1
ENV PROJECT_DIR=$HOME/$PROJECT_NAME

# Set the jupyter user's home directory as the working derectory and switch to the root user.
WORKDIR $HOME
USER root

# Install system dependencies
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    vim \
    less \
    rsync \
    tmux \
    openssh-client \
    gnupg \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    freeglut3-dev \
    libz-dev \
    zlib1g-dev \
    libgl1-mesa-dev \
  && rm -rf /var/lib/apt/lists/*

# Copy configuration files
COPY config_files/overrides.json /opt/conda/share/jupyter/lab/settings/overrides.json
COPY config_files/pycodestyle ./.config/pycodestyle
COPY config_files/mypy_config ./.config/mypy/config

# Copy over project
COPY ./ $PROJECT_DIR

# Change ownership to the jupyter user
RUN chown -R $NB_UID:$NB_GID $PROJECT_DIR

# Install packages that need to be installed through conda
# RUN conda install conda-build
# Slycot install is failing for some reason
#RUN conda install -c conda-forge slycot

# Switch to jupyter user
USER $NB_UID

# Install development packages
RUN pip --no-cache-dir install \
    cmake \
    jupyterlab-lsp \
    'python-language-server[all]' \
    pyls-mypy \
    lckr-jupyterlab-variableinspector \
    ipywidgets \
    aquirdturtle_collapsible_headings \
    jupyterlab-spellchecker \
    ipympl \
    blist \
    "six==1.13.0" \
    "boto3==1.4.8"
#    "google-api-python-client==1.7.8" \
#    "google-oauth" \
#    "kubernetes" \
#    "azure-cli-core==2.22.0" \
#    "azure-mgmt-compute==14.0.0" \
#    "azure-mgmt-msi==1.0.0" \
#    "azure-mgmt-network==10.2.0"

# Install external packages as developer.
RUN pip install -e $PROJECT_DIR/external/cw

# Install topone as developer
RUN pip install -e $PROJECT_DIR

USER root
