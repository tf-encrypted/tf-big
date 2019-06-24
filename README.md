## TF Big

## Developer Requirements

**Linux**

Docker we provide a devel container for tf-big

**MacOS**

libgmp installed

```
brew install gmp
```

## Building

### Tests

**Linux**

```
sudo docker run -it -v `pwd`:/opt/my-project \
  -w /opt/my-project \
  tf-encrypted/tf-big:0.1.0 /bin/bash -c "make test"
```

**MacOS**

### Pip Package