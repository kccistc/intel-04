
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

    Rectangle {
        id: rectangle_how_to_pay
        x: 49
        y: 37
        width: 982
        height: 1842
        opacity: 0.3
        color: "#ffffff"


    }

    Image {
        id: card
        x: 207
        y: 389
        width: 666
        height: 609
        source: "images/card.png"
        fillMode: Image.PreserveAspectFit
    }

    Text {
        id: text1
        x: 322
        y: 1269
        text: qsTr("카드를 삽입해 주세요")
        font.pixelSize: 50
    }

    Text {
        id: text2
        x: 425
        y: 1367
        text: qsTr("초 후에 초기화 됩니다.")
        visible: false
        font.pixelSize: 25
    }

    Timer {
        id: popTimer
        interval: 1000 // 3000 밀리초 = 3초
        repeat: true
        running: true
        property int elapsedSeconds: 6
        onTriggered: {
            popTimer.elapsedSeconds-=1
            text2.text = popTimer.elapsedSeconds + "초 후에 초기화 됩니다."
            text2.visible = true
            if (popTimer.elapsedSeconds < 1){
                while (stackView.depth > 1) {
                    stackView.pop()
                }
            }
        }
    }
}
