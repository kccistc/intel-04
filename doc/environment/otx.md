# OTX (OpenVINO Training eXtensions)

## Reference
[OpenVINO Training Extension - 1.5.0](https://github.com/openvinotoolkit/training_extensions/tree/releases/1.5.0)

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
sudo apt list python3.10-venv
python3.10 -m venv .otx

# Activate virtual env.
source .otx/bin/activate
```

```bash
pip install wheel setuptools

pip install otx==1.5.0rc1 \
            datumaro~=1.5.1rc4 \
            psutil==5.9.* \
            torch==2.0.1 \
            torchvision==0.15.2 \
            addict \
            prettytable==3.9.* \
            mmcv-full==1.7.0 \
            multiprocess==0.70.* \
            pytorchcv==0.0.67 \
            timm==0.6.12 \
            mmcls==0.25.0 \
            mmdeploy==0.14.0 \
            nncf==2.6.0 \
            onnx==1.16.0 \
            openvino-model-api==0.1.9 \
            openvino==2023.0 \
            openvino-dev==2023.0 \
            openvino-telemetry==2023.2.* \
            scipy==1.10.* \
            mmdet==2.28.1 \
            future \
            tensorboard
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

### Practice #1 - Classification
 - [Link](https://openvinotoolkit.github.io/training_extensions/1.5.0/guide/tutorials/base/how_to_train/classification.html)

### Practice #2 - Object Detection
 - [Link](https://openvinotoolkit.github.io/training_extensions/1.5.0/guide/tutorials/base/how_to_train/detection.html)
