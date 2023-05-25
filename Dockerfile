FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
################################################

# Install Python 3.8 and other dependencies
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y build-essential \
    python3.8 python3-setuptools python3.8-dev python3-pip

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
################################################

# Copy app
COPY . /app
################################################

# Install requirements
RUN pip install --no-cache-dir --upgrade torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install .
