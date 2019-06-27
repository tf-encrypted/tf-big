FROM tensorflow/tensorflow:custom-op

RUN wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
RUN tar -xf gmp-6.1.2.tar.xz
RUN cd gmp-6.1.2 && ./configure --enable-cxx --disable-static && make && make check && make install