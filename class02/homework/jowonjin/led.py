from iotdemo.factory_controller import FactoryController

class Outputs:
    BEACON_RED = 2
    BEACON_ORANGE = 3
    BEACON_GREEN = 4

# FactoryController 인스턴스 생성
ctrl = FactoryController('/dev/ttyACM0', debug=True)

# 초기 LED 상태 설정
red_status = ctrl.red
orange_status = ctrl.orange
green_status = ctrl.green

# 시스템 시작
ctrl.system_start()

try:
    while True:
        ans = input("Input Number (1: Start, 2: Stop, 3: Toggle Red, 4: Toggle Orange, 5: Toggle Green, exit: Exit) : ")

        if ans == '1':
            ctrl.system_start()
            print("System Start")

        elif ans == '2':
            ctrl.system_stop()
            print("System Stop")

        elif ans == '3':
            red_status = not red_status
            ctrl.red = red_status
            print(f"Red {'ON' if red_status else 'OFF'}")

        elif ans == '4':
            orange_status = not orange_status
            ctrl.orange = orange_status
            print(f"Yellow {'ON' if orange_status else 'OFF'}")

        elif ans == '5':
            green_status = not green_status
            ctrl.green = green_status
            print(f"Green {'ON' if green_status else 'OFF'}")

        elif ans.lower() == "exit":
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # 시스템 종료
    ctrl.system_stop()
    ctrl.close()
    print("System stopped and closed.")
