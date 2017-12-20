#FROM ubuntu:16.04
FROM caffe2ai/caffe2:c2v0.8.1.cuda8.cudnn7.ubuntu16.04
MAINTAINER Naren Dasan <sivagna2@illinois.edu>

# caffe2 install with cpu support

########## INSTALLATION STEPS ###################
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        ca-certificates \
        wget \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        vim \
        emacs24 \
        nano \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install requests

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        bash_kernel \
        && \
python -m bash_kernel.install \
        && \
python -m ipykernel.kernelspec

EXPOSE 8888

RUN mkdir /opt/cluster-container
RUN cd /opt/cluster-container

COPY start-jupyter.sh /opt/cluster-container
COPY jupyter_get.py /opt/cluster-container
RUN mkdir /examples
COPY mnist.py /examples
COPY jupyter_notebook_config.py /root/.jupyter/
RUN chmod +x /opt/cluster-container/start-jupyter.sh

WORKDIR "/vault"

CMD ["/opt/cluster-container/start-jupyter.sh",  "--allow-root"]


