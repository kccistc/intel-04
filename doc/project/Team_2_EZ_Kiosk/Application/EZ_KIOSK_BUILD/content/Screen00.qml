

/*
This is a UI file (.ui.qml) that is intended to be edited in Qt Design Studio only.
It is supposed to be strictly declarative and only uses a subset of QML. If you edit
this file manually, you might introduce QML code that is not supported by Qt Design Studio.
Check out https://doc.qt.io/qtcreator/creator-quick-ui-forms.html for details on .ui.qml files.
*/
import QtQuick 6.2
import QtQuick.Controls 6.2
import QtQuick.Controls.Material 6.2
import EZ_KIOSK_CLEAN

Rectangle {
    id: rectangle
    width: 1070
    height: 1900
    color: "#f09d27"

    property string fh_or_tg: "initial"
    Image {
        id: image
        x: 0
        y: 0
        width: 1080
        height: 1398
        source: "images/burger_logo.jpg"
        cache: false
        fillMode: Image.PreserveAspectFit

        Button {
            id: button_start
            x: 197
            y: 1377
            width: 699
            height: 302
            text: qsTr("주문 시작")
            font.pointSize: 30
            font.family: "Pretendard"
            onClicked:
                stackView.push(Qt.resolvedUrl(("Screen01.qml")))


            Connections {
                target:server_signal
                onBwChanged: {
                    stackView.push(Qt.resolvedUrl(("Screen02_01.qml")))
                    timer.start()
                }
            }
        }


        // Timer를 사용하여 지연을 추가
        Timer {
            id:timer
            interval: 1000  // 1000 밀리초 = 1초
            repeat: false
            onTriggered: {
                RPi_signal.setBW(true)
                server_signal.setVS(true)
            }
        }

        Text {
            id: text1
            text: "Result from C++: " + connectEvent.result
            anchors.verticalCenter: button_start.verticalCenter
            anchors.left: button_start.right
            anchors.right: button_start.left
            anchors.top: button_start.bottom
            anchors.bottom: button_start.top
            anchors.leftMargin: -492
            anchors.rightMargin: -448
            anchors.topMargin: -409
            anchors.bottomMargin: 93
            font.pixelSize: 12
            anchors.horizontalCenter: button_start.horizontalCenter
            visible: false
        }
    }
}
