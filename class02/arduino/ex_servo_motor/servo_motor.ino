#include <Servo.h>    // 서보모터 라이브러리

Servo servo;    // 서보모터 사용을 위한 객체 생성

int motor = 2;  // 서보모터의 핀 
int angle = 90; // 서보모터 초기 각도 값
void setup() {
  servo.attach(motor);  // 서보모터 연결
  Serial.begin(9600);  // 시리얼 모니터 시작
    
  Serial.println("Enter the u or d"); // u 또는 d키 입력하기
  Serial.println("u = angle + 30");   // u를 누른다면 현재 각도값에서 +30도
  Serial.println("d = angle - 30\n");   // d를 누른다면 현재 각도값에서 -30도
}

void loop() {
  if(Serial.available())  // 시리얼모니터가 사용가능할 때
  {
    char input = Serial.read(); // 문자 입력받기
    
    if(input == 'u')    // u 키를 누를 때
    {
      Serial.print("+30");  // '+30'를 시리얼 모니터에 출력
      for(int i = 0; i < 30; i++)  // 현재 각도에서 15도 더해주기
      {
        angle = angle + 1;   
        if(angle >= 180)
          angle = 180;
                    
        servo.write(angle); 
        delay(10);
      }
      Serial.print("\t\t");
      Serial.println(angle);  // 현재 각도 출력
    } 
    else if(input == 'd')   // 'd'키를 입력했을 때
    {
      Serial.print("\t-30\t");  // '-30'라고 출력
      for(int i = 0 ; i < 30 ; i++)  // 현재 각도에서 30도 빼주기
      {
        angle = angle - 1;
        if(angle <= 0)
          angle = 0;
        servo.write(angle);
        delay(10);
      }
      Serial.println(angle);  // 현재 각도 출력
    }
    else  // 잘못된 문자열을 입력했을 때
    {
      Serial.println("wrong character!!");
    }
  }
}
