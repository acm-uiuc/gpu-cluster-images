FROM pytorch-cudnnv6

MAINTAINER Naren Dasan <sivagna2@illinois.edu>

WORKDIR /workspace
RUN chmod -R a+w /workspace
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
        && \
python -m ipykernel.kernelspec

COPY jupyter_notebook_config.py /root/.jupyter/

EXPOSE 8888

RUN mkdir /notebooks
WORKDIR "/notebooks"

RUN rm /bin/sh &&\
ln -s /bin/bash /bin/sh

COPY start-jupyter.sh /
RUN chmod +x /start-jupyter.sh

CMD ["/start-jupyter.sh",  "--allow-root"]


