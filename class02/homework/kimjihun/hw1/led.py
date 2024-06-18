from iotdemo.factory_controller import FactoryController



ctrl = FactoryController('/dev/ttyACM0')


while True:
    num = int(input('숫자: '))

    if num == 1:
        ctrl.system_start()
    elif num == 2:
        ctrl.system_stop()
    elif num == 3:
        ctrl.red = ctrl.red
    elif num == 4:      
        ctrl.orange = ctrl.orange
    elif num == 5:
        ctrl.green = ctrl.green
    elif num == 6:
        ctrl.conveyor = ctrl.conveyor
    elif num == 7:
        ctrl.push_actuator(1)
    elif num == 8:
        ctrl.push_actuator(2)
    elif num == 9:
        break

ctrl.close()


