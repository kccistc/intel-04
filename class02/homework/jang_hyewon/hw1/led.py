from iotdemo import FactoryController
ctrl = FactoryController('/dev/ttyACM0')

while True:
    commend = input(': ')

    if commend == "1":
        ctrl.system_start()
        print("system_start")
        
    elif commend == "2":   
        ctrl.system_stop()
        print("system_stop")
        break
    
    elif commend == "3":
        ctrl.red = ctrl.red
        print("red")
        
    elif commend == "4":
        ctrl.orange = ctrl.orange
        print("orange")
        
    elif commend == "5":
        ctrl.green = ctrl.green
        print("green")
        
    elif commend == "6":
        ctrl.green = ctrl.conveyor
        print("conveyor")
        
    elif commend == "7":
        ctrl.push_actuator(1)
        print("push_actuator 1")
        
    elif commend == "8":
        ctrl.push_actuator(2)
        print("push_actuator 2")
            
ctrl.close()          
