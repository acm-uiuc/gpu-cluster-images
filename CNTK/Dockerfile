FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04


LABEL maintainer "Naren Dasan <sivagna2@illinois.edu" \
      com.microsoft.cntk.version="2.3"

ENV CNTK_VERSION="2.3"

RUN apt-get update && apt-get install -y --no-install-recommends \
    # General
        ca-certificates \
        wget \
        && \
    # Clean-up
    apt-get -y autoremove \
        && \
    rm -rf /var/lib/apt/lists/*

# Get CNTK Binary Distribution
RUN CNTK_VERSION_DASHED=$(echo $CNTK_VERSION | tr . -) && \
    CNTK_SHA256="013d1050f2f8d7240274e140f41b01a29508f2296388728aababb776fef58667" && \
    wget -q https://cntk.ai/BinaryDrop/CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    echo "$CNTK_SHA256 CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz" | sha256sum --check --strict - && \
    tar -xzf CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    rm -f CNTK-${CNTK_VERSION_DASHED}-Linux-64bit-GPU.tar.gz && \
    /bin/bash /cntk/Scripts/install/linux/install-cntk.sh --py-version 35 --docker

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

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

#RUN ln -s /usr/bin/python3 /usr/bin/python

COPY jupyter_notebook_config.py /root/.jupyter/

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# TensorBoard
EXPOSE 6006
# IPython
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