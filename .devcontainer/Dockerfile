FROM mcr.microsoft.com/devcontainers/miniconda:0-3

USER root

# Update system and install ffmpeg
RUN sudo apt update && \
    sudo apt install --no-install-recommends -y \
        ffmpeg \
        libsm6 \
        libxext6 && \
    sudo apt clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# # Copy contents
COPY requirements.txt /usr/src/app/requirements.txt

# # Install pip packages
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt
# after numpy build
RUN pip install --no-cache lap  

COPY . /usr/src/app  

ENV OMP_NUM_THREADS=1
