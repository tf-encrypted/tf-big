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

```
sudo docker run -it -v `pwd`:/opt/my-project \
  -w /opt/my-project \
  tf-encrypted/tf-big:0.1.0 /bin/bash
```

### Tests

### Pip Package