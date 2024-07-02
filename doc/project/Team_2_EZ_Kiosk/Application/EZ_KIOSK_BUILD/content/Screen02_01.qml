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
        id: normal_bag
        x: 0
        y: 1277
        width: 1080
        height: 502
        color: "#ffffff"

        Flickable {
            id: flickable
            x: 0
            y: 0
            width: 1080
            height: 500
            contentWidth: parent.width * 3
            contentHeight: parent.height
            boundsBehavior: Flickable.StopAtBounds
            interactive: true

            Rectangle {
                id: contentItem
                height: parent.height
                width: childrenRect.width
                color: "#ffffff"
                visible: true

                Row {
                    id: itemRow
                    x: 0
                    spacing: 10
                    anchors.verticalCenter: parent.verticalCenter
                    signal updateParentAmount(int dumb)
                    property int amount_val: 0

                    onUpdateParentAmount: {
                        amount_val = 0
                        for (var i = 0; i < itemRow.children.length; i++) {
                            var child = itemRow.children[i]
                            if (child.hasOwnProperty("sub_total")) {
                                amount_val += child.sub_total
                            }
                        }
                        text1.text = "총액: "+amount_val
                    }
                }
            }
        }
    }

    Item {
        id: item_burger
        x: 0
        y: 184
        width: 1080
        height: 317

        Label {
            id: label_normal_burger
            x: 70
            y: 15
            text: qsTr("햄버거")
            font.pointSize: 16
            font.family: "Pretendard"
        }

        Repeater {

            id: normal_burger_repeater

            model: 4
            Rectangle {
                id: normal_burger
                x: (index * 250) + 70
                y: 70
                width: 200
                height: 250
                color: "#ffffff"

                Image {
                    id: normal_burger_img
                    anchors.fill: parent
                    verticalAlignment: Image.AlignTop
                    source: burgers_db.burgers[index].burger_image
                    sourceSize.height: 170
                    sourceSize.width: 200
                    fillMode: Image.PreserveAspectFit
                }

                Text {
                    id: normal_burger_name
                    x: 0
                    y: 175
                    width: 200
                    height: 33
                    text: burgers_db.burgers[index].burger_name
                    horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                    font.pointSize: 10
                    color: "black"
                }


                Connections{
                    target: server_signal
                    onBagSignal: {
                        handleBagSignal(item_name)
                    }
                }

                // JavaScript 함수 정의
                function handleBagSignal(item_name) {

                    if (item_name === burgers_db.burgers[index].burger_name) {
                        burger_img_button.clicked();
                    }
                }


                Text {
                    id: normal_burger_price
                    x: 0
                    y: 210
                    width: 200
                    height: 40
                    text: burgers_db.burgers[index].burger_price
                    horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                    font.pointSize: 10
                    color: "black"
                }

                Button {
                    id: burger_img_button
                    x: 0
                    y: 0
                    width: 200
                    height: 250
                    opacity: 0
                    onClicked: {
                        enabled = false
                        var newItem = Qt.createQmlObject(
                                    'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 350; height: 350; color: "white"; property int b_counter: 1; property int sub_total: burgers_db.burgers[index].burger_price_int;
Image {
id: normal_burger_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: burgers_db.burgers[index].burger_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit

Component.onCompleted:{
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_burger_name
x: 70
y: 330
width: 200
height: 33
text: burgers_db.burgers[index].burger_name +" "+ burgers_db.burgers[index].burger_price_int+"원"
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: burger_count_row
x: 0
y: 370
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
b_counter-= 1
sub_total = b_counter * burgers_db.burgers[index].burger_price_int
itemRow.updateParentAmount(1)
if (b_counter < 1){
b_counter = 0
normal_burger_repeater.itemAt(index).children[3].enabled=true
parent.parent.destroy()
}

}
font.pointSize: 10
}

Text {
id: countText
text: b_counter
}
Button {
text: "+"
onClicked: {
b_counter+=1
sub_total = b_counter * burgers_db.burgers[index].burger_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}', itemRow)
                        if (newItem === null) {
                            console.log("Error creating object")
                        } else {
                            newItem.parent = itemRow
                        }
                    }
                    Connections {
                        target: burger_img_button

                    }
                }
            }
        }
    }

    Item {
        id: item_chicken
        x: 0
        y: 535
        width: 1080
        height: 317

        Label {
            id: label_normal_chicken
            x: 70
            y: 15
            text: qsTr("치킨")
            font.pointSize: 16
            font.family: "Pretendard"
        }

        Repeater {
            id: normal_chicken_repeater
            model: 4
            Rectangle {
                id: normal_chicken
                x: (index * 250) + 70
                y: 70
                width: 200
                height: 250
                color: "#ffffff"

                Image {
                    id: normal_chicken_img
                    anchors.fill: parent
                    verticalAlignment: Image.AlignTop // 이미지가 Rectangle을 채우도록 설정
                    source: chickens_db.chickens[index].chicken_image // 이미지 경로 설정
                    fillMode: Image.PreserveAspectFit // 이미지가 비율을 유지하며 Rectangle을 채우도록 설정
                }

                Text {
                    id: normal_chicken_name
                    x: 0
                    y: 171
                    width: 200
                    height: 33
                    text: chickens_db.chickens[index].chicken_name
                    horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                    font.pointSize: 10
                    color: "black"
                    }


                Connections{
                    target: server_signal
                    onBagSignal: {
                        handleBagSignal(item_name)
                    }
                }

                // JavaScript 함수 정의
                function handleBagSignal(item_name) {

                    if (item_name === chickens_db.chickens[index].chicken_name) {
                        chicken_img_button.clicked();
                    }
                }



                Text {
                    id: normal_chicken_price
                    x: 0
                    y: 210
                    width: 200
                    height: 40
                    text: chickens_db.chickens[index].chicken_price
                    horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                    font.pointSize: 10
                    color: "black"
                }

                Button {
                    id: chicken_img_button
                    text: ""
                    x: 0
                    y: 0
                    width: 200
                    height: 250
                    opacity: 0
                    onClicked: {
                        enabled = false
                        var newItem = Qt.createQmlObject(
                                    'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 350; height: 350; color: "white"; property int c_counter: 1; property int sub_total: chickens_db.chickens[index].chicken_price_int;
Image {
id: normal_chicken_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: chickens_db.chickens[index].chicken_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit
Component.onCompleted:{
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_chicken_name
x: 70
y: 330
width: 200
height: 33
text: chickens_db.chickens[index].chicken_name +" "+ chickens_db.chickens[index].chicken_price_int
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: chicken_count_row
x: 0
y: 370
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
c_counter-= 1
sub_total = c_counter * chickens_db.chickens[index].chicken_price_int
itemRow.updateParentAmount(1)
if (c_counter < 1){
c_counter = 0
normal_chicken_repeater.itemAt(index).children[3].enabled=true
parent.parent.destroy()
}
}
font.pointSize: 10
}

Text {
id: countText
text: c_counter
}
Button {
text: "+"
onClicked: {
c_counter+= 1
sub_total = c_counter * chickens_db.chickens[index].chicken_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}', itemRow)

                        if (newItem === null) {
                            console.log("Error creating object")
                        } else {
                            newItem.parent = itemRow
                        }
                    }

                    Connections {
                        target: chicken_img_button

                    }
                }
            }
        }
    }
    Item {
        id: item_beverage
        x: 0
        y: 887
        width: 1080
        height: 317


        Label {
            id: label_normal_beverage
            x: 70
            y: 15
            text: qsTr("음료수")
            font.pointSize: 16
            font.family: "Pretendard"
        }

        Repeater {
            id: normal_beverage_repeater
            model: 4
            Rectangle {
                id: normal_beverage
                x: (index * 250) + 70
                y: 70
                width: 200
                height: 250
                color: "#ffffff"

                Image {
                    id: normal_beverage_img
                    anchors.fill: parent
                    verticalAlignment: Image.AlignTop // 이미지가 Rectangle을 채우도록 설정
                    source: beverages_db.beverages[index].beverage_image // 이미지 경로 설정
                    fillMode: Image.PreserveAspectFit // 이미지가 비율을 유지하며 Rectangle을 채우도록 설정
                }

                Text {
                    id: normal_beverage_name
                    x: 0
                    y: 171
                    width: 200
                    height: 33
                    text: beverages_db.beverages[index].beverage_name
                    horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                    font.pointSize: 10
                    color: "black"
                }

                Connections{
                    target: server_signal
                    onBagSignal: {
                        handleBagSignal(item_name)
                    }
                }

                // JavaScript 함수 정의
                function handleBagSignal(item_name) {

                    if (item_name === beverages_db.beverages[index].beverage_name) {
                        beverage_img_button.clicked();
                    }
                }


                Text {
                    id: normal_beverage_price
                    x: 0
                    y: 210
                    width: 200
                    height: 40
                    text: beverages_db.beverages[index].beverage_price
                    horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                    font.pointSize: 10
                    color: "black"
                }

                Button {
                    id: beverage_img_button
                    text: ""
                    x: 0
                    y: 0
                    width: 200
                    height: 250
                    opacity: 0
                    onClicked: {
                        enabled = false
                        var newItem = Qt.createQmlObject(
                                    'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 350; height: 350; color: "white"; property int be_counter: 1; property int sub_total: beverages_db.beverages[index].beverage_price_int;
Image {
id: normal_beverage_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: beverages_db.beverages[index].beverage_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit
Component.onCompleted:{
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_beverage_name
x: 70
y: 330
width: 200
height: 33
text: beverages_db.beverages[index].beverage_name +" "+ beverages_db.beverages[index].beverage_price_int + "원"
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: beverage_count_row
x: 0
y: 370
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
be_counter-= 1
sub_total = be_counter * beverages_db.beverages[index].beverage_price_int
itemRow.updateParentAmount(1)
if (be_counter < 1){
be_counter = 0
normal_beverage_repeater.itemAt(index).children[3].enabled=true
parent.parent.destroy()
}
}
font.pointSize: 10
}

Text {
id: countText
text: be_counter
}
Button {
text: "+"
onClicked: {
be_counter+= 1
sub_total = be_counter * beverages_db.beverages[index].beverage_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}', itemRow)

                        if (newItem === null) {
                            console.log("Error creating object")
                        } else {
                            newItem.parent = itemRow
                        }
                    }

                    Connections {
                        target: beverage_img_button

                    }
                }
            }
        }
    }
    Rectangle {
        id: normal_buttons
        x: 0
        y: 1779
        width: 1080
        height: 141
        color: "#f5efb3"
        border.width: 0

        Connections{
            target: server_signal
            onCompleteSignal: {
                normal_button_pay.clicked()
                normal_pay_yes.clicked()
            }
        }

        Button {
            id: normal_button_pay
            x: 0
            y: 0
            width: 1080
            height: 141
            text: qsTr("결제")
            font.pointSize: 30
            font.family: "Pretendard"
            Material.background: Material.Amber
            onClicked: {
                normal_modalPopup.visible = true
            }
        }
    }
    Button {
        id: normal_button_pop
        x: 371
        y: 32
        width: 338
        height: 124
        text: qsTr("초기 화면으로")
        Material.background: Material.Amber
        font.bold: true
        font.pointSize: 18
        font.family: "Pretendard"

        Connections {
            target: normal_button_pop
            onClicked: {
                stackView.pop()
            }
        }
    }
    Popup {
        id: normal_modalPopup
        x: 290
        y: 880
        width: 500
        height: 200
        font.pointSize: 11
        font.family: "Pretendard"
        modal: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        contentItem: Rectangle {
            color: "lightgray"
            border.color: "black"
            anchors.fill: parent

            Column {
                anchors.centerIn: parent
                spacing: 10

                Label {
                    text: "결제 하시겠습니까?"
                    horizontalAlignment: Text.AlignHCenter
                    font.pointSize: 18
                    font.family: "Pretendard"
                }
                Row {
                    anchors.horizontalCenter: parent.horizontalCenter
                    spacing: 10

                    Button {
                        id: normal_pay_yes
                        text: "네"
                        font.pointSize: 10
                        font.family: "Pretendard"
                        onClicked: {
                            normal_modalPopup.visible = false
                            var total_amount = itemRow.amount_val
                            stackView.push(Qt.resolvedUrl(
                                                       ("Screen03.qml")))
                        }

                    }
                    Button {
                        id: normal_pay_no
                        text: "아니오"
                        font.pointSize: 10
                        font.family: "Pretendard"
                        onClicked: normal_modalPopup.visible = false
                    }
                }
            }
        }
    }

    Text {
        id: text1
        x: 31
        y: 1320
        text: qsTr("총액:")
        font.pixelSize: 25
    }
}
