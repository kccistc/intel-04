# Project Counting Cars~

* 이 프로젝트는 고속도로 CCTV 영상 내 교통량을 자동으로 측정하는 시스템이다.

## Architecture

* 고속도로에 차 지나다니는 동영상 입력 -> YOLOv8 이용한 차량 tracking -> 가상의 선을 지나면 차량 Count -> 차종 별 Count 수 집계

## Members

* 신나라
* 허찬욱

## Used

* YOLOv8, OpenVINO의 Person-Counting-Webcam

## Steps to run

```shell
cd ~/xxxx
source .venv/bin/activate

cd /intel-04/class01/mini-project/07_Chanuk,Nara/mini_prj_chanuk,nara
python3 CountingCars2.py
```

## Output
![image](https://github.com/kccistc/intel-04/assets/169637084/8e2a8940-6e5a-4680-b1d5-4602520cd6ca)


## Appendix
* (참고 자료 및 알아두어야할 사항들 기술)
