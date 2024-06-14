from iotdemo import FactoryController

ctrl = FactoryController('/dev/ttyACM0')

while True:
    input_num = input("input number: ")

    if input_num =='1':
        ctrl.system_start()
    
    elif input_num =='2':
         ctrl.system_stop()

    elif input_num == '3':
         ctrl.red = ctrl.red
    
    elif input_num =='4':
         ctrl.orange = ctrl.orange
    
    elif input_num == '5':
        ctrl.green = ctrl.green

    elif input_num =='6':
         ctrl.conveyor = ctrl.conveyor

    elif input_num =='7':
        ctrl.push_actuator(1)
    elif input_num =='8':
        ctrl.push_actuator(2)

    elif input_num =='exit':
        ctrl.close()
        break
