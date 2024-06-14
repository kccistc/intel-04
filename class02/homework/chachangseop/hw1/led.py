"""
아두이노 LED 과제
"""

from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

ctrl.system_start()

while True:
    answer = int(input("숫자 입력하세요 (1 ~ 8 | 0 : exit) : "))
    if answer == 0:
        break
    elif answer == 1:
        ctrl.system_start()
        print("start")
    elif answer == 2:
        ctrl.system_stop()
        print("stop")
    elif answer == 3:
        ctrl.red = ctrl.red
        print("red")
    elif answer == 4:
        ctrl.orange = ctrl.orange
        print("orange")
    elif answer == 5:
        ctrl.green = ctrl.green
        print("green")
    elif answer == 6:
        ctrl.conveyor = ctrl.conveyor
        print("conveyor")
    elif answer == 7:
        ctrl.push_actuator(1)
        print("actuator1")
    elif answer == 8:
        ctrl.push_actuator(2)
        print("actuator2")
    else:
        print("다시 입력하세요")
        continue

ctrl.close()
print("시스템 종료")
