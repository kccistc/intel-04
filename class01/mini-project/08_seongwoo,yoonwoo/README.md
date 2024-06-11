## Project "진실의 눈"

* (간략히 프로젝트를 설명하고, 최종 목표가 무엇인지에 대해 기술)

  
Person tracking으로 사람(객체)을 탐지하고, Mono depth를 이용해 가까운 사람을 찾는 프로젝트 "진실의 눈"
monodepth를 이용하여 카메라(진실의 눈)로 가까운 사람을 탐지하고, 센터 좌표를 이용해 어떠한 객체가 앞서있는지 판단하는 것이 최종 목표이다.

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
