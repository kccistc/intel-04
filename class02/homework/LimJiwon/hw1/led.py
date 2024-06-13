from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

while True:
    input_num = input('Enter num >>> ')

    if input_num == '1':
        ctrl.system_start()
        print("system start")

    elif input_num == '2':   
        ctrl.system_stop()
        print("system stop")

    elif input_num == '3':
        ctrl.red = ctrl.red
        print("red")

    elif input_num == '4':
        ctrl.orange = ctrl.orange
        print("orange")

    elif input_num == '5':
        ctrl.green = ctrl.green
        print("green")

    elif input_num == '6':
        ctrl.conveyor = ctrl.conveyor
        print("conveyor")

    elif input_num == '7':
        ctrl.push_actuator(1)
        print("push actuator 1")

    elif input_num == '8':
        ctrl.push_actuator(2)
        print("push actuator 2")

    elif input_num == '0':
        break

ctrl.close() 
