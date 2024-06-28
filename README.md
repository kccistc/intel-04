# 상공회의소 서울기술교육센터 인텔교육 4기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kccistc/intel-04.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team projects

### 제출방법

1. 팀구성 및 프로젝트 세부 논의 후, 각 팀은 프로젝트 진행을 위한 Github repository 생성

2. [doc/project/README.md](./doc/project/README.md)을 각 팀이 생성한 repository의 main README.md로 복사 후 팀 프로젝트에 맞게 수정 활용

3. 과제 제출시 `인텔교육 4기 Github repository`에 `New Issue` 생성. 생성된 Issue에 하기 내용 포함되어야 함.

    * Team name : Project Name
    * Project 소개
    * 팀원 및 팀원 역활
    * Project Github repository
    * Project 발표자료 업로드

4. 강사가 생성한 `Milestone`에 생성된 Issue에 추가 

### 평가방법

* [assessment-criteria.pdf](./doc/project/assessment-criteria.pdf) 참고

### 제출현황

### Team: 뭔가 센스있는 팀명
<프로젝트 요약>
* Members
  | Name | Role |
  |----|----|
  | 채치수 | Project lead, 프로젝트를 총괄하고 망하면 책임진다. |
  | 송태섭 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 정대만 | UI design, 사용자 인터페이스를 정의하고 구현한다. |
  | 채소연 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 권준호 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |
* Project Github : https://github.com/goodsense/project_awesome.git
* 발표자료 : https://github.com/goodsense/project_aewsome/doc/slide.ppt


### Team: 쿡포유 (cook for you) - 1조
<프로젝트 요약>

"냉장고에 있는 재료에 맞춘 레시피의 추천과 레시피를 알려주는 프로그램"

* Members
  | Name | Role |
  |----|----|
  | 장혜원 | Project Leader, DB engineer, UI engineer  |
  | 강동혁 | UI engineer   |
  | 김민정 | OCR AI modeling, Project Manager |
  | 유광균 | Computer Vision AI modeling |
  
* Project Github : https://github.com/Intel-I4/ingredient_check
* 발표자료 : https://github.com/kccistc/intel-04/blob/main/doc/project/Team_I4/I4-presentation.pptx



### Team: E-Z Kiosk(가칭) - 2조
<프로젝트 요약>

"누군가에게는 어려운 키오스크를 누구나 쉽게 사용하는 키오스크로"

* Members
  | Name | Role |
  |----|----|
  | 염재영 | Project lead(프로젝트 관리), UI design  |
  | 김기범 | Computer Vision AI modeling, NLP modeling   |
  | 조원진 | Computer Vision AI modeling, NLP modeling  |
  | 정용재 | Project Manager(서비스 기획), UI engineer, AI 모델 검증|
  
* Project Github : https://github.com/wodud6423/Intel-Edge-AI-.git
* 발표자료 : https://github.com/wodud6423/Intel-Edge-AI-/tree/main


### Team: 쿠크루 (Cupid Crew) - 3조
<프로젝트 요약>

"OpenVINO를 이용한 홈트레이닝"

* Members
  | Name | Role |
  |----|----|
  | 신나라 | Project lead(프로젝트 관리), UI design  |
  | 김지훈 | Computer Vision AI modeling, NLP modeling   |
  | 임지원 | Computer Vision AI modeling, NLP modeling  |
  | 차창섭 | Project Manager(서비스 기획), UI engineer, AI 모델 검증|
  
* Project Github : https://github.com/sournara/Intel-Edge-AI-Project
* 발표자료 : https://github.com/kccistc/intel-04/blob/main/doc/project/Team3/team3.pdf


### Team:   드가자(드론으로부터 가드하는 자들)  - 4조

<프로젝트 요약>

“카메라와 모터를 활용하여 민간에서도 사용가능한 드론추적시스템을 개발”

* Members


    | Name | Role |
    | --- | --- |
    | 이성우 | Project Lead(프로젝트 관리), Board Support Package(보드 제어)  |
    | 김민정A | Data Processing , AI 모델 검증 |
    | 김윤우 | Board Support Package(보드 제어),  Image Processing(영상 처리)  |
    | 허찬욱  | Project Manager(서비스 기획), Computer Vision AI Modeling |


* Project Github : https://github.com/kyoonw/D-gaja-.git
* 발표자료 : https://github.com/kyoonw/D-gaja-/tree/main

### Team: 이츠유얼턴 - 5조
<프로젝트 요약>

"비보호 좌회전 및 유턴 안전 보조 시스템"

* Members
  | Name | Role |
  |----|----|
  | 김승민 | Project lead, UI design  |
  | 박민혁 | Project Manager, AI modeling    |
  | 유지승 | AI modeling, Hardware Setup   |
  | 조수환 | AI modeling, System integration |

* Project Github : https://github.com/suhwanjo/Intel-Eged-AI-Project.git
* 발표자료 : https://github.com/wodud6423/Intel-Edge-AI-/tree/main

### Team: 고픈비노 - 6조
<프로젝트 요약>
"사용자 인식과 손인식을 통한 모니터 위치 자동 조절"
* Members
  | Name | Role |
  |----|----|
  | 김기훈 | Project lead(프로젝트 관리), H/W engineer |
  | 이지원 | AI modeling |
  | 정희중 | AI modeling |
  | 최재혁 | H/W engineer |

* Project Github : https://github.com/KORIDO1234/gopenvino.git
* 발표자료 : https://github.com/KORIDO1234/gopenvino/tree/main

### Team: 가오가이거 - 7조

<프로젝트 요약>
”벤치프레스 사고 예방 알림 시스템"

- Members
    
    
    | Name | Role |
    | --- | --- |
    | 김정대 | Project lead(프로젝트 관리), 프로젝트를 총괄 |
    | 권오준 | AI modeling, UI designer |
    | 이윤혁 | AI modeling, H/W engineer |
- Project Github : https://github.com/jd6286/openvino_project
- 발표자료 : 추후 작성

