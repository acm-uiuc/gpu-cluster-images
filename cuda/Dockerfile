FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER Naren Dasan <sivagna2@illinois.edu>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        rsync \
        software-properties-common \
        unzip \
        vim \
        emacs24 \
        nano \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT /bin/bash
