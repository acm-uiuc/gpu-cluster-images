FROM pytorch-cudnnv6

MAINTAINER Naren Dasan <sivagna2@illinois.edu>

WORKDIR /workspace
RUN chmod -R a+w /workspace

RUN apt update && apt -y install \
        vim \
        emacs24 \
        nano \
       && \
       apt-get clean && \
       rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        torchvision \
        bash_kernel \
        && \
python -m bash_kernel.install \
        && \
python -m ipykernel.kernelspec

COPY jupyter_notebook_config.py /root/.jupyter/

EXPOSE 8888

RUN mkdir /notebooks
WORKDIR "/notebooks"

COPY start-jupyter.sh /
COPY jupyter_get.py /
RUN mkdir /examples
COPY mnist.py /examples
RUN chmod +x /start-jupyter.sh

CMD ["/start-jupyter.sh",  "--allow-root"]


