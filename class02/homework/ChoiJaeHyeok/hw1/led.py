from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

while True:
    input_key = input('input key (1~8) :')

    if input_key == '1':
        ctrl.system_start()
        print("system_start")

    elif input_key == '2':
        ctrl.system_stop()
        print("system_stop")
        ctrl.close()
        break

    elif input_key == '3':
        ctrl.red = ctrl.red
        print("red")

    elif input_key == '4':
        ctrl.orange = ctrl.orange
        print("orange")

    elif input_key == '5':
        ctrl.green = ctrl.green
        print("green")

    elif input_key == '6':
        ctrl.conveyor = ctrl.conveyor
        print("conveyor")

    elif input_key == '7':
        ctrl.push_actuator(1)
        print("push_acuator1")

    elif input_key == '8':
        ctrl.push_actuator(2)
        print("push_acuator2")
