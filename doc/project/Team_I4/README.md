# 쿡포유 (cook for you)
냉장고에 있는 재료에 맞춘 레시피의 추천과 레시피를 알려주는 프로그램 
</br></br></br>


## Git clone 
```
git clone https://github.com/Intel-I4/ingredient_check.git
```

</br>


## 시장 조사
#### 30인을 대상으로 여론조사 실시</br>
![전체설문조사](https://cdn.discordapp.com/attachments/1245893304118415425/1256048162586497087/image.png?ex=667f5982&is=667e0802&hm=e822cef5fdf68271b5480dadc10407ad165a978a15aa2670e9a0d56be306b891&)   
* [그림 1] 전체 30인 중 요리함 : 13명, 요리안함 : 17명</br></br>
* [그림 2] 요리를 하는 사람 13인을 대상으로 음식을 만들 때 가장 신경쓰는 부분을 질문한 결과, 재료라는 응답이 61.5% </br></br>
* [그림 3] 모든 사람을 대상으로 레시피를 찾는 장소를 질문한 결과, 가장 많은 사람이 찾는 사이트는 유튜브라는 응답이 28명 (중복응답)</br></br>
* [그림 4] 모든 사람들 대상으로 레시피의 재료를 준비하는 방법에 대한 질문 결과, 레시피를 그대로 준비한다는 응답과 냉장고에 남은 재료를 조합해가면서 요리한다는 응답이 각각 44.2%와 41.9%로 과반수 이상을 차지 (중복응답)</br>

</br>


## High Level Design

<p align="center">
  <img src="https://file.notion.so/f/f/6cbc4593-1f87-4219-99f0-e012e89996a2/5f168b95-bd5c-426d-a20d-627736b2feb1/Untitled.png?id=67c31580-329b-4c04-bbc5-6ebbd4d879ea&table=block&spaceId=6cbc4593-1f87-4219-99f0-e012e89996a2&expirationTimestamp=1719626400000&signature=ZvTvRTR8tayTwbmkpvYh39ugJPWyFMh58_U1CrRfLdg&downloadName=Untitled.png">
</p>



</br>


## Prerequite
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


</br>


## Steps to run

- (프로젝트 실행방법에 대해서 기술, 특별한 사용방법이 있다면 같이 기술)

```
cd ~/xxxx
source .venv/bin/activate

python3 main.py
```


</br>


## Outout

<p align="center">
  <img src="https://file.notion.so/f/f/6cbc4593-1f87-4219-99f0-e012e89996a2/31899a91-0f42-431a-9e4f-e05bdf142f94/Screenshot_from_2024-07-02_16-30-52.png?id=e9aca1d4-cfba-468a-bbd1-199dc0d10a03&table=block&spaceId=6cbc4593-1f87-4219-99f0-e012e89996a2&expirationTimestamp=1719993600000&signature=LLqGIY8aF1wybKKwwhqKCIKD4yGf0LtiaAYn54W0lyM&downloadName=Screenshot+from+2024-07-02+16-30-52.png">
</p>


</br>




## 참고

[일정 및 일지](https://jang-hw.notion.site/I4-2024-372453cc120247ff860577e3eaf6c50d?pvs=74)
[레시피 크롤링 참고 사이트](https://otugi.tistory.com/393)   


