# TF Big

TF Big adds big number support to TensorFlow, allowing computaitons to be performed on arbitrary precision integers. Internally these are represented as variant tensors of [GMP](https://gmplib.org/) values, exposed in Python through the `tf_big.Tensor` wrapper for convenience. For importing and exporting, numbers are typically expressed as strings.

[![PyPI](https://img.shields.io/pypi/v/tf-big.svg)](https://pypi.org/project/tf-big/) [![CircleCI Badge](https://circleci.com/gh/tf-encrypted/tf-big/tree/master.svg?style=svg)](https://circleci.com/gh/tf-encrypted/tf-big/tree/master)

## Usage

```python
import tensorflow as tf
import tf_big

# load large values as strings
x = tf_big.constant(["100000000000000000000", "200000000000000000000"])

# load ordinary TensorFlow tensors
y = tf_big.convert_to_tensor(tf.constant([3, 4]))

# perform computation as usual
z = x * y

# use TensorFlow sessions to evalutate the results
with tf.Session() as sess:
  res = sess.run(z)
  print(res)
```

## Installation

Python 3 packages are available from [PyPI](https://pypi.org/project/tf-big/):

```
pip install tf-big
```

See below for further instructions for setting up a development environment.

## Development

### Requirements

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) to set up and use a Python 3.5 or 3.6 environment for all instructions below:

```
conda create -n tfbig-dev python=3.6
source activate tfbig-dev
```

#### Ubuntu

The only requirement for Ubuntu is to have [docker installed](https://docs.docker.com/install/linux/docker-ce/ubuntu/). This is the recommended way to [build custom operations for TensorFlow](https://github.com/tensorflow/custom-op). We provide a custom development container for TF Big with all dependencies already installed.

#### macOS

Setting up a development environment on macOS is a little more involved since we cannot use a docker container. We need four things:

- Python (>= 3.5)
- [Bazel](https://www.bazel.build/) (>= 0.15.0)
- [GMP](https://gmplib.org/) (>= 6.1.2)
- [TensorFlow](https://www.tensorflow.org/) (== 1.13.1)

Using [Homebrew](https://brew.sh/) we first make sure that both [Bazel](https://docs.bazel.build/versions/master/install-os-x.html#install-with-installer-mac-os-x) and GMP are installed:

```
brew tap bazelbuild/tap
brew install bazelbuild/tap/bazel
brew install gmp
brew install mmv
```

The remaining PyPI packages can then be installed using:

```
pip install -r requirements-dev.txt
```

### Testing

#### Ubuntu

Run the tests on Ubuntu by running the `make test` command inside of a docker container. Right now, the docker container doesn't exist on docker hub yet so we must first build it:

```
docker build -t tf-encrypted/tf-big:0.1.0 .
```

Then we can run `make test`:

```
sudo docker run -it \
  -v `pwd`:/opt/my-project -w /opt/my-project \
  tf-encrypted/tf-big:0.1.0 /bin/bash -c "make test"
```

#### macOS

Once the development environment is set up we can simply run:

```
make test
```

This will install TensorFlow if not previously installed and build and run the tests.

### Building pip package

#### macOS

```
make build
```
