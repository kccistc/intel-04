# OTX (OpenVINO Training eXtensions)

## Install dependencies
```
sudo apt update
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev gcc-multilib dkms mesa-utils
sudo apt upgrade
```

## Download and install CUDA 11.7
Note, link below is for Ubuntu20.04. For other versoins please refer [CUDA Toolkit 11.7 Downloads](https://developer.nvidia.com/cuda-11-7-0-download-archive)
```bash
cd ~/Downloads/
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run
```


## OTX install
<!-- Source code install
```bash
mkdir -p ~/repo && cd $_
git clone https://github.com/openvinotoolkit/training_extensions.git
cd training_extensions
git checkout develop
```
-->


```bash
# Create virtual env.
python3 -m venv .otx

# Activate virtual env.
source .otx/bin/activate
```

```bash
pip install wheel setuptools

# install command for torch==1.13.1 for CUDA 11.7:
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install otx[full]
```

## OpenVINO hello_classification
```
https://raw.githubusercontent.com/openvinotoolkit/openvino/master/samples/python/hello_classification/hello_classification.py
```

## Dataset

A command combination that shows number of files under given directory.
```bash
find ./ -maxdepth 2 –type d | while read –r dir; do printf “%s:\t” “$dir”; find “$dir” –type f | wc –l; done
```

### Links
* Flowers:
    http://download.tensorflow.org/example_images/flower_photos.tgz

* Dogs & cats:
    https://drive.google.com/file/d/1_ItEl2QLWhYtaTeyOP90jCZmB2ofcQW0/view?usp=drive_link

