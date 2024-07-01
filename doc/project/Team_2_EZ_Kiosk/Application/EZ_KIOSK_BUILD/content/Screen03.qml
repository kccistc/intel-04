
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

    Connections{
        target: server_signal
        onChoiceSignal: {
            handleChoiceSignal(method_name)
        }
    }

    // JavaScript 함수 정의
    function handleChoiceSignal(method_name) {

        if (method_name === "card") {
            button_card.checked = true
        }else if (method_name === "cash") {
            button_cash.checked = true
        }else if (method_name === "for_here") {
            for_here.checked = true
        }else if (method_name === "to_go") {
            to_go.checked = true
        }else if (method_name === "good"){
            feedback_button_good.checked = true
            button_next.clicked()
        }else if (method_name === "bad"){
            feedback_button_bad.checked = true
            button_next.clicked()
        }
    }


    Rectangle {
        id: rectangle_how_to_pay
        x: 49
        y: 37
        width: 982
        height: 1842
        opacity: 0.3
        color: "#ffffff"

    }

    Row {
        id: pay_method
        x: 81
        y: 288
        width: 919
        height: 238
        spacing: 29



        Button {
            id: button_card
            width: 450
            height: 250
            text: qsTr("카드")
            checkable: true
            font.pointSize: 30
            font.family: "Pretendard"
            Material.background: Material.Amber
            autoExclusive: true
        }

        Button {
            id: button_cash
            width: 450
            height: 250
            text: qsTr("현금")
            checkable: true
            font.pointSize: 30
            font.family: "Pretendard"
            autoExclusive: true
            Material.background: Material.Amber
        }
    }
    Row {
        id: eating_method
        x: 81
        y: 748
        width: 919
        height: 238
        spacing: 29
        Button {
            id: for_here
            width: 450
            height: 250
            text: qsTr("먹고가기")
            checkable: true
            font.pointSize: 30
            font.family: "Pretendard"
            autoExclusive: true
            Material.background: Material.Amber
        }

        Button {
            id: to_go
            width: 450
            height: 250
            text: qsTr("포장하기")
            checkable: true
            font.pointSize: 30
            font.family: "Pretendard"
            autoExclusive: true
            Material.background: Material.Amber
        }
    }

    Row {
        id: feedback_buttons
        x: 81
        y: 1208
        width: 919
        height: 238
        spacing: 29
        Button {
            id: feedback_button_good
            width: 450
            height: 250
            text: qsTr("만족해요")
            checkable: true
            font.pointSize: 20
            font.family: "Pretendard"
            autoExclusive: true
            Material.background: Material.Amber
        }

        Button {
            id: feedback_button_bad
            width: 450
            height: 250
            text: qsTr("별로에요")
            checkable: true
            font.pointSize: 20
            font.family: "Pretendard"
            autoExclusive: true
            Material.background: Material.Amber
        }
    }

    Button {
        id: button_next
        x: 326
        y: 1625
        width: 428
        height: 167
        text: qsTr("다음")
        font.pointSize: 20
        font.family: "Pretendard"
        Material.background: Material.Amber

        Connections {
            target: button_next
            onClicked: {
                if (button_card.checked && (for_here.checked || to_go.checked))
                    stackView.push(Qt.resolvedUrl(("Screen04.qml")))
                else if (button_cash.checked && (for_here.checked
                                                 || to_go.checked))
                    stackView.push(Qt.resolvedUrl(("Screen05.qml")))

                server_signal.setVS(false)


            }
        }
    }
    Text {
        id: text1
        x: 119
        y: 197
        text: qsTr("결제 방법")
        font.pixelSize: 25
    }

    Text {
        id: text2
        x: 119
        y: 686
        text: qsTr("식사 장소")
        font.pixelSize: 25
    }

    Text {
        id: text3
        x: 119
        y: 1142
        text: ""
        font.pixelSize: 25

        Component.onCompleted: {
            if (server_signal.getVS()) {
                text3.text = "음성 서비스 만족도"
            } else {
                text3.text = "서비스 만족도"
                }
            }


    }
}
