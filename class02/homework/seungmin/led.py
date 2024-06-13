#! /src/bin/ python3.8
"""led.py"""
from iotdemo import FactoryController


def main():
    """main"""
    ctrl = FactoryController('/dev/ttyACM0')

    if ctrl.is_dummy:
        print("Connection Fail!!")
        return


    while True:
        num = input("\n1. system_start()\
            \n2. system_stop()\
            \n3. red\
            \n4. orange\
            \n5. green\
            \n6. conveyor\
            \n7. push_actuator(1)\
            \n8. push_actuator(2)\
            \n9. exit\n\n")

        if num == "1":
            ctrl.system_start()
        elif num == "2":
            ctrl.system_stop()
        elif num == "3":
            if ctrl.red:
                ctrl.red = True
            else:
                ctrl.red = False
        elif num == "4":
            if ctrl.orange:
                ctrl.orange = True
            else:
                ctrl.orange = False
        elif num == "5":
            if ctrl.green:
                ctrl.green = True
            else:
                ctrl.green = False
        elif num == "6":
            if ctrl.conveyor:
                ctrl.conveyor = True
            else:
                ctrl.conveyor = False
        elif num == "7":
            ctrl.push_actuator(1)
        elif num == "8":
            ctrl.push_actuator(2)
        elif num == "9":
            ctrl.close()
            break


if __name__ == "__main__":
    main()
