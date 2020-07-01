FROM tensorflow/tensorflow:nightly-custom-op-ubuntu16

# Install tools needed for building
RUN apt update && \
    apt install -y \
        curl git python3 tree rsync mmv \
        pkg-config g++ cmake \
        zip unzip zlib1g-dev

# Install Python versions needed
RUN curl -OL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -f && \
    ~/miniconda3/bin/conda create -n py3.5 python=3.5 -y && \
    ln -s ~/miniconda3/envs/py3.5/bin/python ~/python3.5 && \
    ~/miniconda3/bin/conda create -n py3.6 python=3.6 -y && \
    ln -s ~/miniconda3/envs/py3.6/bin/python ~/python3.6

# Install extra dependencies; in case of TF Big this is just GMP
RUN wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz && \
    tar -xf gmp-6.1.2.tar.xz && \
    cd gmp-6.1.2 && \
    ./configure --with-pic --enable-cxx --enable-static --disable-shared && \
    make && \
    make check && \
    make install && \
    make clean
