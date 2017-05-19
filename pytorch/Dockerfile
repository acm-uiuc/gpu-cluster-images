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
        && \
python -m ipykernel.kernelspec

EXPOSE 8888

RUN mkdir /notebooks
WORKDIR "/notebooks"

#CMD ["jupyter notebook \"$@\""]
