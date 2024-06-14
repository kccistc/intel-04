from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

while True:
    input_num = input('enter input(q: exit) : ')

    if input_num == '1':
        ctrl.system_start()
        print("system_start") // 시스템 실행 

    elif input_num == '2':
        ctrl.system_stop()
        print("system_stop") // 시스템 정지

    elif input_num == '3':
        ctrl.red = ctrl.red
        print("red") // 빨간 다이오드 

    elif input_num == '4':
        ctrl.orange = ctrl.orange
        print("orange") // 노란 다이오드

    elif input_num == '5':
        ctrl.green = ctrl.green
        print("green") // 초록 다이오드

    elif input_num == '6':
        ctrl.conveyor = ctrl.conveyor
        print("conveyor")

    elif input_num == '7':
        ctrl.push_actuator(1)
        print("push_actuator1")

    elif input_num == '8':
        ctrl.push_actuator(2)
        print("push_actuator2")

    elif input_num == 'q':
        print("exit")
        break

ctrl.close()
