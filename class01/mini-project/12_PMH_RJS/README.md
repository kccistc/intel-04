# Project "알려줄카"

* (간략히 프로젝트를 설명하고, 최종 목표가 무엇인지에 대해 기술)<br>
비보호 좌회전을 위한 미니 프로젝트입니다. 신호 대기 중에 반대편 차량을 detection하고, monodepth를 통해 인식된 차량과의 거리를 측정하여 일정 거리 내에 들어오면 경고 알림을 제공하는 것이 이번 프로젝트의 목표입니다. 

## 프로젝트 PPT 링크

* https://www.canva.com/design/DAGH4Ok6OGw/kd8TlSgfW2OkH_tmzbcN7Q/view?utm_content=DAGH4Ok6OGw&utm_campaign=share_your_design&utm_medium=link&utm_source=shareyourdesignpanel

## High Level Design

* (프로젝트 아키텍쳐 기술, 전반적인 diagram 으로 설명을 권장)

## Clone code

* (각 팀에서 프로젝트를 위해 생성한 repository에 대한 code clone 방법에 대해서 기술)

```shell
git clone https://github.com/xxx/yyy/zzz
```

## Prerequite

* (프로잭트를 실행하기 위해 필요한 dependencies 및 configuration들이 있다면, 설치 및 설정 방법에 대해 기술)

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Steps to build

* (프로젝트를 실행을 위해 빌드 절차 기술)

```shell
cd ~/xxxx
source .venv/bin/activate

make
make install
```

## Steps to run

* (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

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

