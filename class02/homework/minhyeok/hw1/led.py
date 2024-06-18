""" 아두이노 파이썬 프로젝트 """
# import sys
# sys.path.append('/home/ubuntu/my_repo/intel-04/class02/smart-factory')

from iotdemo import FactoryController

print("1")
ctrl = FactoryController('/dev/ttyACM0')
print("2")
ctrl.system_start()
print("3")
while True:
    print("4")
    ans = input("Input Number : ")
    
    if ans == '1':
        ctrl.system_start()
        print("System Start")
        
    elif ans == '2':
        ctrl.system_stop()
        print("System Stop")
        
    elif ans == '3':
        ctrl.red = ctrl.red
        print("Red")
        
    elif ans == '4':
        ctrl.orange = ctrl.orange
        print("Orange")
    
    elif ans == '5':
        ctrl.green = ctrl.green
        print("Green")
        
    elif ans == '6':
        ctrl.conveyor = ctrl.conveyor
        print("conveyor")
        
    elif ans == '7':
        ctrl.push_actuator(1)
        print("actuator1")
        
    elif ans == '8':
        ctrl.push_actuator(2)
        print("actuator2")
        
    elif ans.lower() == "exit":
        break

ctrl.close()
