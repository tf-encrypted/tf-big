## TF Big

TF Big provides some basic operations for big integers. TF Big uses libgmp for its optimized big integer routines.

## Developer Requirements

**Ubuntu**

The only requirement for Ubuntu is to have docker installed. This is the recommended way to build custom operations for tensorflow. Please see the documentation [here](https://github.com/tensorflow/custom-op). We provide a custom development container for TF Big which contains the libgmp dependency already installed.

The documentation for installing docker on Ubuntu can be found [here](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

**MacOS**

TODO simplify this, add a bootstrap script to Makefile

Since we can't use a MacOS docker container, setting up a development environment is a little more involved. We need four things:

- Python 3.5 or 3.6
- Homebrew
- Bazel 0.15.0
- libgmp
- Tensorflow 1.13.1 **TODO** support 1.14, might be a little involved

We recommend using [Anaconda](https://www.anaconda.com/distribution/) to set up a Python 3.5 or 3.6 environment. Once Anaconda is installed this can be done with:

```
$ conda create -n py36 python=3.6
$ source activate py36
```

We recommed using [Homebrew](https://brew.sh/) to install the next couple of dependencies. This can be installed easily with:

```
$ /usr/bin/ruby -e "$(curl -fsSL \
    https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Bazel recommends installing with their binary installed. The documentation for this can be found [here](https://docs.bazel.build/versions/master/install-os-x.html#install-with-installer-mac-os-x). But if you have Homebrew already installed you can install bazel with a couple of simple commands:

```
$ brew tap bazelbuild/tap
$ brew install bazelbuild/tap/bazel
```

Next, we can install libgmp with Homebrew:

```
brew install gmp
```

Tensorflow will be installed automatically when using the Makefile so no need to install it manually but it can be done before hand by using pip:

```
pip install tensorflow==1.13.1
```

## Building

### Tests

**Ubuntu**

Run the tests on Ubuntu by running the `make test` command inside of a docker container. Right now, the docker container doesn't exist on docker hub yet so we must first build it:

```
docker build -t tf-encrypted/tf-big:0.1.0 .
```

Then we can run `make test`:

```
sudo docker run -it -v `pwd`:/opt/my-project \
  -w /opt/my-project \
  tf-encrypted/tf-big:0.1.0 /bin/bash -c "make test"
```

**MacOS**

Once the environment is set up we can simply run:

```
make test
```

This will install Tensorflow if not previously installed and build and run the tests.

### Pip Package

TODO