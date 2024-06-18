void setup() {
  // put your setup code here, to run once:
  pinMode(7, OUTPUT);
  pinMode(5, INPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  if(digitalRead(5)==HIGH){
    digitalWrite(7, HIGH);
  }
  delay(20);
  digitalWrite(7, LOW);
}

