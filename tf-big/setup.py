"""Installing with setuptools."""
import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="tf-big",
    version="0.1.0",
    packages=setuptools.find_packages(),
    package_data={'tf_big': []},
    python_requires=">=3.5",
    install_requires=[
        "tensorflow >=1.13.1, <2",
    ],
    extras_require={
        "tf": ["tensorflow>=1.12.0,<2"],
    },
    license="Apache License 2.0",
    url="https://github.com/tf-encrypted/tf-big",
    description="Arbitrary precision integers in TensorFlow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="The TF Encrypted Authors",
    author_email="tfencrypted@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Development Status :: 2 - Pre-Alpha",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ]
)
