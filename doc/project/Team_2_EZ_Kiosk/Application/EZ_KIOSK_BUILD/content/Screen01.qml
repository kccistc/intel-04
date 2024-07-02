

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

    Column {
        id: column
        x: -27
        y: 117
        width: 921
        height: 475
        spacing: 140

        Button {
            id: button_order_normal
            x: 166
            y: 94
            width: 808
            height: 450
            Material.background: Material.Amber
            text: qsTr("일반주문")
            font.bold: false
            font.pointSize: 25
            font.family: "Pretendard"
            onClicked: fh_or_tg = "for_here"

            Connections {
                target: button_order_normal
                onClicked: {
                    stackView.push(Qt.resolvedUrl(("Screen02_01.qml")))
                }
            }
        }

        Button {
            id: button_order_easy
            x: 166
            y: 703
            width: 808
            height: 450
            Material.background: Material.Amber
            text: qsTr("쉬운주문")
            font.bold: false
            font.pointSize: 25
            font.family: "Pretendard"
            onClicked: fh_or_tg = "to_go"

            Connections {
                target: button_order_easy
                onClicked: {
                    stackView.push(Qt.resolvedUrl(("Screen02_02.qml")))
                }
            }
        }

        Button {
            id: button_order_voice
            x: 166
            y: 1285
            width: 808
            height: 450
            text: qsTr("음성주문")
            font.pointSize: 25
            font.family: "Pretendard"
            font.bold: false
            Connections {
                target: button_order_voice
                onClicked: {
                    voiceEvent.vflag = 1
                    stackView.push(Qt.resolvedUrl(("Screen02_01.qml")))
                    server_signal.setVS(true)
                }
            }
            Material.background: Material.Amber
        }

    }

    Text {
        id: text2
        x: 214
        y: 1629
        text: "vFlag from C++: " + voiceEvent.vflag
        font.pixelSize: 20
        visible: false
    }
}
