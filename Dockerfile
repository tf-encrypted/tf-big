FROM tensorflow/tensorflow:custom-op

RUN wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
RUN tar -xf gmp-6.1.2.tar.xz
RUN cd gmp-6.1.2 && ./configure --with-pic --enable-cxx --enable-static --disable-shared && make && make check && make install

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -f

SHELL ["/bin/bash", "-c"]

RUN /root/miniconda3/bin/conda init bash

# how to make conda init changes take effect below
RUN /root/miniconda3/bin/conda create -n py35 python=3.5 -y
RUN /root/miniconda3/bin/conda create -n py36 python=3.6 -y
RUN /root/miniconda3/envs/py35/bin/python -m venv venv-py35 && . venv-py35/bin/activate && pip install tensorflow==1.13.1
RUN /root/miniconda3/envs/py36/bin/python -m venv venv-py36 && . venv-py36/bin/activate && pip install tensorflow==1.13.1
