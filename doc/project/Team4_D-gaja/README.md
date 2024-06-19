# D-gaja

* (간략히 프로젝트를 설명하고, 최종 목표가 무엇인지에 대해 기술)

## High Level Design
![image](https://github.com/kyoonw/D-gaja/assets/169637084/314f441b-72fb-4a20-bd46-a6295f902bf9)


## Clone code

* github를 통한 코드 공유 

```shell
git clone https://github.com/kyoonw/D-gaja.git
```

## Prerequite

* requirements.txt를 통해 파이썬 환경설치

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Steps to build

* pc가상환경을 통해 testbuild 후 board-depoly

```shell
cd ~/xxxx
source .venv/bin/activate

make
make install
```

## Steps to run

* pc가상환경을 통해 test build 후 board-depoly

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
