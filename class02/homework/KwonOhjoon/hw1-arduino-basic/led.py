from iotdemo import FactoryController


ctrl = FactoryController('/dev/ttyACM0')

try:
    # 초기화
    ctrl.red = True
    ctrl.orange = True
    ctrl.green = True
    ctrl.conveyor = False

    # 터미널에 번호 입력 시 API 호출
    while True:
        value = int(input("Enter the value >> "))

        if value == 1:
            # system_start
            pass
        elif value == 2:
            # system_stop
            pass
        elif value == 3:
            ctrl.red = ctrl.red
            pass
        elif value == 4:
            ctrl.orange = ctrl.orange
            pass
        elif value == 5:
            ctrl.green = ctrl.green
            pass
        elif value == 6:
            ctrl.conveyor = ctrl.conveyor
            pass
        elif value == 7:
            ctrl.push_actuator(1)
            pass
        elif value == 8:
            # push_actuator(2)
            ctrl.push_actuator(2)
        else:
            print("Invalid value")
except KeyboardInterrupt:
    pass
finally:
    ctrl.red = True        # LED는 True일 때 불이 꺼지네?
    ctrl.orange = True
    ctrl.green = True
    ctrl.conveyor = False

ctrl.close()
