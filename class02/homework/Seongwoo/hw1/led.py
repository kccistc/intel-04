"""Arduino LED Project + init, exit 기능추가본"""
from iotdemo import FactoryController


ctrl = FactoryController('/dev/ttyACM0')

ctrl.red = ctrl.DEV_OFF
ctrl.orange = ctrl.DEV_OFF
ctrl.green = ctrl.DEV_OFF
ctrl.conveyor = ctrl.DEV_ON


while True:
    input_num = input(" input pin number: ")
    if input_num == '1':
        ctrl.system_start()
        print("Start")
    elif input_num == '2':
        ctrl.system_stop()
        print("Stop")
    elif input_num == '3':
        ctrl.red = ctrl.red
        print("red")
    elif input_num == '4':
        ctrl.orange = ctrl.orange
        print("orange")
    elif input_num == '5':
        ctrl.green = ctrl.green
        print("orange")
    elif input_num == '6':
        ctrl.conveyor = ctrl.conveyor
        print("conveyor")
    elif input_num == '7':
        ctrl.push_actuator(1)
        print("push_actuator-1")
    elif input_num == '8':
        ctrl.push_actuator(2)
        print("push_actuator-2")
    elif input_num == '0':
        print("init")
        ctrl.red = ctrl.DEV_OFF
        ctrl.orange = ctrl.DEV_OFF
        ctrl.green = ctrl.DEV_OFF
        ctrl.conveyor = ctrl.DEV_ON
    elif input_num == 'exit':
        ctrl.close()
        break
