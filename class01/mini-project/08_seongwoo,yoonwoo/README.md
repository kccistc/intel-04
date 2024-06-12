## Member
김윤우, 이성우


## Project Name & Purpose

  
Person tracking으로 사람(객체)을 탐지하고, Monodepth를 이용해 가장 가까운 사람을 찾는 프로젝트 "황야의 무법자 - 영희"


monodepth를 이용하여 웹캠으로 가까운 사람을 탐지하고, 센터 좌표를 이용해 어떠한 객체가 가장 앞서있는지 근접도를 판별하는 것이 최종 목표이다.


## Used Model

* Person Tracking, Monodepth 사용


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

![image](https://github.com/kccistc/intel-04/assets/170689181/7dd53fa5-ed79-487f-9d0d-4503de5b902e)


## Appendix

* 사진 출처 : 넷플릭스 오리지널 시리즈 '오징어 게임'. 넷플릭스 제공
