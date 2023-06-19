FROM streamlit-yolov8:latest


# Update system and install ffmpeg
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        ffmpeg \
        libsm6 \
        libxext6 \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY requirements.txt /usr/src/app/requirements.txt

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt


COPY . /usr/src/app  

ENV OMP_NUM_THREADS=1

