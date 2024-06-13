# Project 요약해VOCA

* Open Model Zoo와 다른 여러 pre-trained 모델을 활용하여 사용자가 업로드한 문서의 텍스트를 인식(OCR), 요약(LLM), 음성 변환(TTS)하는 애플리케이션을 개발하는 것을 목표로 합니다. 이를 통해 사용자는 문서의 핵심 내용을 빠르게 파악하고 음성으로 들을 수 있는 편리한 솔루션을 제공합니다.

## High Level Design

### 프로젝트 아키텍처

* 프로젝트는 세 개의 주요 모듈로 구성됩니다:

    1. OCR 모듈: Tesseract OCR을 사용하여 이미지에서 텍스트를 추출합니다.
    2. 텍스트 요약 모듈: Transformers 라이브러리의 BERT 모델을 사용하여 텍스트를 요약합니다.
    3. TTS 모듈: Open Model Zoo의 ForwardTacotron 및 WaveRNN 모델을 사용하여 요약된 텍스트를 음성으로 변환합니다.

### 다이어그램

![Screenshot_from_2024-06-13_09-26-59](https://github.com/suhwanjo/Intel-Edge-AI-mini-project/assets/112834460/6a83560e-423b-4eb2-bc18-f8c9db55310f)

## Clone code

```shell
git clone https://github.com/suhwanjo/Intel-Edge-AI-mini-project.git
```

## Prerequite

```shell
cd ~/Intel-Edge-AI-mini-project/cd mini_project2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
omz_downloader --list models_tts.lst
omz_downloader --list models_ocr.lst
```

## Steps to run

```shell
cd ~/Intel-Edge-AI-mini-project/mini_project2
source .venv/bin/activate
python3 app.py
```

## Output

![Screenshot from 2024-06-13 09-35-08](https://github.com/suhwanjo/Intel-Edge-AI-mini-project/assets/112834460/f249ba38-c559-41f5-abab-61ea4af6ce35)
![Screenshot from 2024-06-13 09-37-32](https://github.com/suhwanjo/Intel-Edge-AI-mini-project/assets/112834460/9ec4a2f1-94b2-4263-aa8e-e8ed0c442488)
```shell
apple
is
embracing
generative
ai
buzzy
form
of
intelligence
that
can
provide
and
thorough
to
the
companys
with
a
essentially
turning
it
into
an
iphone
chatbot
this
siri
to
perform
specific
such
as
recalling
a
ago
on
the
or
answering
the
weather
the
news
or
it
can
perform
more
advanced
as
answering
a
is
landing
by
analyzing
information
sent
in
an
email
over
it
learn
the
preferences
and
respond
this
is
not
some
competitors
have
already
introduced
in
generative
siri
will
also
automatically
and
to
users
based
on
audio
and
natural
language
with
images
and
contextual
cues
```
![Screenshot from 2024-06-13 09-38-27](https://github.com/suhwanjo/Intel-Edge-AI-mini-project/assets/112834460/9d02c82a-7521-4690-b9f2-fa60d8cba2bb)
