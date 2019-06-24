## TF Big

TF Big implements some basic operations on big integers. TF Big uses [libgmp](https://gmplib.org/) for its optimized big integer routines.

## Requirements

For Linux, the recommended way to build and run tests is using docker. Please these instructions for installing docker on Ubuntu.

libgmp needs to be installed prior to building TF Big. libgmp is provided simply through [Homebrew](https://brew.sh/) and through apt for Ubuntu. Install using one of the following methods:

**MacOS**

```
brew install gmp
```

**Ubuntu**

```
sudo apt install gmp
```

## Building

### Tests

**Linux**

```
sudo docker run -it -v `pwd`:/opt/my-project \
  -w /opt/my-project \
  tensorflow/tensorflow:custom-op \
  /bin/bash -c "sudo apt install gmp && make test"
```

**MacOS**

```
make test
```

### Pip Package

TODO

