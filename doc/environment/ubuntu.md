# Ubuntu Environment Setup

## Ubuntu Install

* [Ubuntu22.04 Web Page](https://releases.ubuntu.com/jammy/)

* [rufus Tool](https://rufus.ie/ko/)

## Basic Packages

```bash
$ sudo apt install -y build-essential \
    software-properties-common \
    vim \
    terminator \
    gcc \
    git \
    git-all \
    make \
    cmake \
    htop \
    net-tools \
    tree \
    mplayer \
    mesa-utils \
    intel-opencl-icd \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv
```

## Visual Studio Code

```bash
$ sudo apt update
$ sudo apt install -y software-properties-common \
	apt-transport-https \
	wget

$ wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
$ sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
$ sudo apt install -y code

$ code
```

## Python Install

```bash
$ sudo add-apt-repository ppa:deadsnakes/ppa
$ sudo apt update

$ sudo apt install -y python3.8 \
	python3.8-venv \
	python3.8-dev \
	python3.8-distutils

$ python3.8
```

## Kernel Version Change

```bash
$ wget https://raw.githubusercontent.com/pimlie/ubuntu-mainline-kernel.sh/master/ubuntu-mainline-kernel.sh
```
