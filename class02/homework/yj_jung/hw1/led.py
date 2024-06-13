from iotdemo.factory_controller import FactoryController


def control_arduino():

    ctrl = FactoryController('/dev/ttyACM0')

    try:
        while True:
            input_cmd = int(input('숫자입력 : '))

            if input_cmd == 1:
                ctrl.system_start()
            elif input_cmd == 2:
                ctrl.system_stop()
            elif input_cmd == 3:
                ctrl.red = ctrl.red
            elif input_cmd == 4:
                ctrl.orange = ctrl.orange
            elif input_cmd == 5:
                ctrl.green = ctrl.green
            elif input_cmd == 6:
                ctrl.conveyor = ctrl.conveyor
            elif input_cmd == 7:
                ctrl.push_actuator(1)
            elif input_cmd == 8:
                ctrl.push_actuator(2)
            else:
                pass

    except KeyboardInterrupt:
        print('\nExit')

    finally:
        ctrl.close()


if __name__ == "__main__":
    control_arduino()
