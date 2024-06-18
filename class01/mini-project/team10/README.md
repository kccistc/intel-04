## Team members

1. 김정대
2. 권오준
3. 이윤혁

## Purpose

떠오르는 아이디어가 있지만 도면이나 그림으로 바로 나타내기 어려운 사람이 손쉽게 모델링하여 확인할 수 있도록 함

## High Level Design

- 스케치한 도면과 키워드를 입력
    - 도면과 키워드에 따라 새로운 이미지 생성
- 생성된 이미지를 객체만 추출하여 3D 모델링
- 추가로 음악 키워드 입력시 원하는 분위기에 맞는 BGM 생성

![순서도](https://cdn.discordapp.com/attachments/1244871418609664067/1250356660569112596/sh2gh.drawio.png?ex=666aa4e2&is=66695362&hm=6979a748e677ca8dd277be0a91342be57de3b1fc324984a587cde75234cf8196&)

## Github link

https://github.com/jd6286/SH2GH

## Prerequite

```python
python3 -m venv sh2gh
source sh2gh/bin/activate
git clone https://github.com/GaParmar/img2img-turbo.git
# Install requirements
pip3 install -U pip
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Model Excute

```python
python main.py
```

## Result
- GUI
  
![GUI](https://media.discordapp.net/attachments/1244871418609664067/1250262847338844221/program_ui.png?ex=666a4d83&is=6668fc03&hm=1da050c1142f4dc543c4fa1759e366e3ea36f3f9f59ea64ef903223098bec159&=&format=webp&quality=lossless)

- 원본 이미지
  
![original](https://cdn.discordapp.com/attachments/1244871418609664067/1250262846235742228/moai.png?ex=666a4d83&is=6668fc03&hm=fb3c5ca90b5da84b80696f1461d1913db4f4e2ed3cfd4a5542b2cc156a3c1714&)

- 3D 이미지
  
![3dImage](https://media.discordapp.net/attachments/1244871418609664067/1250262846688854026/3d_moai.png?ex=666a4d83&is=6668fc03&hm=2e7d388931d19bca399f1ccda10cb92b545ed36a8bdc0131eb396150d9eba6be&=&format=webp&quality=lossless)

## Demo

https://drive.google.com/file/d/12uHbnKXtz3ksCBNqbf-WtNpigkmD6zYF/view

