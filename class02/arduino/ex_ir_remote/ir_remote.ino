#include <IRremote.h> // ver 3.9.0

int IRsensor = A1;
IRrecv irrecv(IRsensor);
decode_results results;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  irrecv.enableIRIn();
}

void loop() {
  // put your main code here, to run repeatedly:
  if (irrecv.decode(&results)) {
    //Serial.println(results.value, HEX);
    switch (results.value) {
      case 0xFFA25D :
        Serial.println("ch-");
        break;
      case 0xFF629D :
        Serial.println("ch");
        break;
      case 0xFFE21D :
        Serial.println("ch+");
        break;
      case 0xFF22DD :
        Serial.println("<<");
        break;
      case 0xFF02FD :
        Serial.println(">>");
        break;
      case 0xFFC23D :
        Serial.println(">||");
        break;
      case 0xFFE01F :
        Serial.println("-");
        break;
      case 0xFFA857 :
        Serial.println("+");
        break;
      case 0xFF906F :
        Serial.println("EQ");
        break;
      case 0xFF6897 :
        Serial.println("0");
        break;
      case 0xFF9867 :
        Serial.println("100+");
        break;
      case 0xFFB04F :
        Serial.println("200+");
        break;
      case 0xFF30CF :
        Serial.println("1");
        break;
      case 0xFF18E7 :
        Serial.println("2");
        break;
      case 0xFF7A85 :
        Serial.println("3");
        break;
      case 0xFF10EF :
        Serial.println("4");
        break;
      case 0xFF38C7 :
        Serial.println("5");
        break;
      case 0xFF5AA5 :
        Serial.println("6");
        break;
      case 0xFF42BD :
        Serial.println("7");
        break;
      case 0xFF4AB5 :
        Serial.println("8");
        break;
      case 0xFF52AD :
        Serial.println("9");
        break;
    }
    irrecv.resume();
  }
  delay(100);
}
