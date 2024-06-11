# Project SmartScanner

* Open Model Zoo의 여러 pre-trained 모델을 활용하여 사용자가 업로드한 문서의 텍스트를 인식(OCR), 요약(LLM), 음성 변환(TTS)하는 애플리케이션을 개발하는 것을 목표로 합니다. 이를 통해 사용자는 문서의 핵심 내용을 빠르게 파악하고 음성으로 들을 수 있는 편리한 솔루션을 제공합니다.

## High Level Design

### 프로젝트 아키텍처

* 프로젝트는 세 개의 주요 모듈로 구성됩니다:

    1. OCR 모듈: Tesseract OCR을 사용하여 이미지에서 텍스트를 추출합니다.
    2. 텍스트 요약 모듈: Transformers 라이브러리의 BERT 모델을 사용하여 텍스트를 요약합니다.
    3. TTS 모듈: Open Model Zoo의 ForwardTacotron 및 WaveRNN 모델을 사용하여 요약된 텍스트를 음성으로 변환합니다.

### 다이어그램

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


