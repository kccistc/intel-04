""" Arduino와 Python을 이용한 LED 제어 """

from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

while True:
    user_input = int(input("Input Number between 1 and 8 : "))

    if user_input == 1:
        ctrl.system_start()
        print("System Start")

    elif user_input == 2:
        ctrl.system_stop()
        print("System Stop")

    elif user_input == 3:
        ctrl.red = ctrl.red
        print("Red")

    elif user_input == 4:
        ctrl.orange = ctrl.orange
        print("Orange")

    elif user_input == 5:
        ctrl.green = ctrl.green
        print("Green")

    elif user_input == 6:
        ctrl.conveyor = ctrl.conveyor
        print("Conveyor")

    elif user_input == 7:
        ctrl.push_actuator(1)
        print("Actuator1")

    elif user_input == 8:
        ctrl.push_actuator(2)
        print("Actuator2")

    else :
        break

ctrl.close()

