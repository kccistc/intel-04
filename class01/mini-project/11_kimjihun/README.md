# Project ABC

use cam capture my body
use pose estimation and makes pose video
finally cat ref_images(whatever u want) , pose video make character video

## High Level Design

use vae model encode to latents , decode from latents (make referencenet , refimage_combined_pose_images)
use image_encoder make clip(cross attention) ref_images with pose seq thats relationships

## Clone code

i dont use it this project

```shell
git clone https://github.com/xxx/yyy/zzz
```

## Prerequite

in the code it have all resource

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Steps to build

Download the necessary models, call the settings, convert them to a format suitable for pytorch, lighten them, and compress them sequentially to build them according to the shape of the input images.

```shell
cd ~/xxxx
source .venv/bin/activate

make
make install
```

## Steps to run

first buy super computer if u cant dont use this code
if u want run this code reduce the width , height , length and u have to fit to code
if u have error delete the code $$skip if not $should 
because the code combined so have to run the all code

```shell
cd ~/xxxx
source .venv/bin/activate

cd /path/to/repo/xxx/
python demo.py -i xxx -m yyy -d zzz
```

## Output

* (프로젝트 실행 화면 캡쳐)

![./result.jpg](./result.jpg)

## Appendix

* (참고 자료 및 알아두어야할 사항들 기술)
