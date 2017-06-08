#FROM ubuntu:16.04
FROM caffe2ai/caffe2:c2.tutorial.0.7.1
MAINTAINER Naren Dasan <sivagna2@illinois.edu>

# caffe2 install with cpu support

########## INSTALLATION STEPS ###################
RUN apt-get install unzip vim -y --no-install-recommends
RUN pip install requests
WORKDIR "/"
RUN rm -rf caffe2
RUN git clone --recursive https://github.com/caffe2/caffe2.git

########## REBUILD ###################
WORKDIR "/caffe2"
RUN make
WORKDIR "/caffe2/build"
RUN make install
ENV PYTHONPATH /usr/local
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
# must force workdir for python modules to run and install models in desired location; ENV has no effect
WORKDIR "/usr/local"
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

########## SETUP TUTORIAL FILES #################
# get model files for Loading Pre-Trained Models
RUN python -m caffe2.python.models.download -i squeezenet
# not installing these by default to keep the image smaller
# RUN python -m caffe2.python.models.download -i bvlc_alexnet
# RUN python -m caffe2.python.models.download -i bvlc_googlenet
# RUN python -m caffe2.python.models.download -i bvlc_reference_caffenet
# rcnn model throws error, so holding off until fixed
# RUN python -m caffe2.python.models.download -i bvlc_reference_rcnn_ilsvrc13
# RUN python -m caffe2.python.models.download -i finetune_flickr_style
# get MNIST dataset for MNIST
WORKDIR "/caffe2/caffe2/python/tutorials"
RUN mkdir tutorial_data && cd tutorial_data
WORKDIR "/caffe2/caffe2/python/tutorials/tutorial_data"
RUN wget "https://s3.amazonaws.com/caffe2/datasets/mnist/mnist.zip"
RUN unzip -d mnist mnist.zip
WORKDIR "/caffe2/caffe2/python/tutorials"


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
COPY mnist.py /
RUN chmod +x /start-jupyter.sh

CMD ["/start-jupyter.sh", "--allow-root"]