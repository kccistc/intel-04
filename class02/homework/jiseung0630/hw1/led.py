from iotdemo import FactoryController


ctrl = FactoryController('/dev/ttyACM0')

while True:
    keyboard_input = input('Enter input:')

    if keyboard_input == '1':
        ctrl.system_start()
        print("Start")

    elif keyboard_input == '2':
        ctrl.system_stop()
        print("Stop")
        break

    elif keyboard_input == '3':
        ctrl.red = ctrl.red
        print("Red")

    elif keyboard_input == '4':
        ctrl.orange = ctrl.orange
        print("Orange")

    elif keyboard_input == '5':
        ctrl.green = ctrl.green
        print("Green")

    elif keyboard_input == '6':
        ctrl.conveyor = ctrl.conveyor
        print("Conveyor")

    elif keyboard_input == '7':
        ctrl.push_actuator(1)
        print("Acuator1")

    elif keyboard_input == '8':
        ctrl.push_actuator(2)
        print("Acuator2")


ctrl.close()
