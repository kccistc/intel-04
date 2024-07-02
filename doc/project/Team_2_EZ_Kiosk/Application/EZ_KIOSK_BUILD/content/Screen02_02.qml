import QtQuick 6.2
import QtQuick.Controls 6.2
import QtQuick.Controls.Material 6.2
import EZ_KIOSK_CLEAN

Rectangle {
    id: rectangle_elder
    width: 1070
    height: 1900
    color: "#f09d27"

    Rectangle {
        id: elder_bag
        x: 0
        y: 1378
        width: 1080
        height: 401
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

            Text {
                id: text1
                x: 36
                y: 24
                text: qsTr("총액:")
                font.pixelSize: 25
            }
        }
    }

    Item {
        id: elder_item_burger
        x: 0
        y: 161
        width: 1080
        height: 441

        property int indexer : 0


        Rectangle {
            id: elder_burger1
            visible: true
            x: 150
            y: 54
            width: 320
            height: 340
            color: "#ffffff"

            Image {
                id: elder_burger_img1
                anchors.fill: parent
                verticalAlignment: Image.AlignTop // 이미지가 Rectangle을 채우도록 설정
                source: burgers_db.burgers[elder_item_burger.indexer*2].burger_image // 이미지 경로 설정
                sourceSize.height: 260
                sourceSize.width: 320
                fillMode: Image.PreserveAspectFit // 이미지가 비율을 유지하며 Rectangle을 채우도록 설정
            }

            Text {
                id: elder_burger_name1
                x: 60
                y: 260
                width: 200
                height: 33
                text: burgers_db.burgers[elder_item_burger.indexer*2].burger_name
                horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                font.pointSize: 10
                color: "black"
            }

            Text {
                id: elder_burger_price1
                x: 60
                y: 298
                width: 200
                height: 40
                text: burgers_db.burgers[elder_item_burger.indexer*2].burger_price
                horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                font.pointSize: 10
                color: "black"
            }


            // JavaScript 함수 정의
            function handleBagSignal(item_name) {

                if (item_name === burgers_db.burgers[elder_item_burger.indexer*2].burger_name) {
                    elder_burger_img_button_1.clicked();
                }
            }

            Connections{
                target: server_signal
                onBagSignal: {
                    handleBagSignal(item_name)
                }
            }

            Button {
                id: elder_burger_img_button_1
                x: 0
                y: 0
                width: 200
                height: 250
                opacity: 0
                onClicked: {
                    var this_index = 0
                    this_index = elder_item_burger.indexer*2
                    var newItem = Qt.createQmlObject(
                                'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 250; height: 200; color: "lightgray"; property int b_counter: 1; property int this_index: 0; property int sub_total: 0;
Image {
id: normal_burger_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: burgers_db.burgers[this_index].burger_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit

Component.onCompleted:{
sub_total = burgers_db.burgers[this_index].burger_price_int
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_burger_name
x: 0
y: 200
width: 250
height: 33
text: burgers_db.burgers[this_index].burger_name +" "+ burgers_db.burgers[this_index].burger_price_int+"원"
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: burger_count_row
x: 70
y: 250
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
b_counter-= 1
sub_total = b_counter * burgers_db.burgers[this_index].burger_price_int
itemRow.updateParentAmount(1)
if (b_counter < 1){
b_counter = 0
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
sub_total = b_counter * burgers_db.burgers[this_index].burger_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}',itemRow)
                    //enabled = false
                    newItem.this_index = this_index
                    if (newItem === null) {
                        console.log("Error creating object")
                    } else {
                        newItem.parent = itemRow
                    }
                }
            }
        }

        Rectangle {
            id: elder_burger2
            visible: true
            x: 550
            y: 54
            width: 320
            height: 340
            color: "#ffffff"

            Image {
                id: elder_burger_img2
                anchors.fill: parent
                verticalAlignment: Image.AlignTop // 이미지가 Rectangle을 채우도록 설정
                source: burgers_db.burgers[elder_item_burger.indexer*2+1].burger_image // 이미지 경로 설정
                fillMode: Image.PreserveAspectFit // 이미지가 비율을 유지하며 Rectangle을 채우도록 설정
                sourceSize.height: 260
                sourceSize.width: 320
            }

            Text {
                id: elder_burger_name2
                x: 60
                y: 260
                width: 200
                height: 33
                text: burgers_db.burgers[elder_item_burger.indexer*2+1].burger_name
                horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                font.pointSize: 10
                color: "black"
            }

            Text {
                id: elder_burger_price2
                x: 60
                y: 298
                width: 200
                height: 40
                text: burgers_db.burgers[elder_item_burger.indexer*2+1].burger_price
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

                if (item_name === burgers_db.burgers[elder_item_burger.indexer*2+1].burger_name) {
                    elder_burger_img_button_2.clicked();
                }
            }


            Button {
                id: elder_burger_img_button_2
                x: 0
                y: 0
                width: 200
                height: 250
                opacity: 0
                onClicked: {
                    var this_index = elder_item_burger.indexer*2+1
                    var newItem = Qt.createQmlObject(
                                'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 250; height: 200; color: "lightgray"; property int b_counter: 1; property int this_index: 0; property int sub_total: 0;
Image {
id: normal_burger_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: burgers_db.burgers[this_index].burger_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit

Component.onCompleted:{
sub_total += burgers_db.burgers[this_index].burger_price_int
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_burger_name
x: 0
y: 200
width: 250
height: 33
text: burgers_db.burgers[this_index].burger_name +" "+ burgers_db.burgers[this_index].burger_price_int+"원"
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: burger_count_row
x: 70
y: 250
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
b_counter-= 1
sub_total = b_counter * burgers_db.burgers[this_index].burger_price_int
itemRow.updateParentAmount(1)
if (b_counter < 1){
b_counter = 0
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
sub_total = b_counter * burgers_db.burgers[this_index].burger_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}',itemRow)
                    //enabled = false
                    newItem.this_index = this_index
                    if (newItem === null) {
                        console.log("Error creating object")
                    } else {
                        newItem.parent = itemRow
                    }
                }
            }
        }

        Label {
            id: elder_label_burger
            x: 74
            y: 8
            text: qsTr("햄버거")
            font.pointSize: 16
            font.family: "Pretendard"
        }

        Button {
            id: elder_refresh_burger
            x: 909
            y: 243
            width: 119
            height: 131
            text: qsTr("다른\n메뉴")
            font.pointSize: 10
            font.family: "Pretendard"
            onClicked:{
                if(elder_item_burger.indexer === 1){
                    elder_item_burger.indexer = 0
                }
                else{
                    elder_item_burger.indexer = 1
                }
            }
        }
    }

    Item {
        id: elder_item_chicken
        x: 0
        y: 551
        width: 1080
        height: 422

        property int indexer : 0

        Label {
            id: elder_label_chicken
            x: 74
            y: 8
            text: qsTr("치킨")
            font.pointSize: 16
            font.family: "Pretendard"
        }

        Rectangle {
            id: elder_chicken1
            x: 147
            y: 54
            width: 320
            height: 340
            color: "#ffffff"
            visible: true

            Image {
                id: elder_chicken1_img
                anchors.fill: parent
                verticalAlignment: Image.AlignTop // 이미지가 Rectangle을 채우도록 설정
                source: chickens_db.chickens[elder_item_chicken.indexer*2].chicken_image // 이미지 경로 설정
                fillMode: Image.PreserveAspectFit // 이미지가 비율을 유지하며 Rectangle을 채우도록 설정
                sourceSize.height: 260
                sourceSize.width: 320
            }

            Text {
                id: elder_chicken1_name
                x: 60
                y: 260
                width: 200
                height: 33
                text: chickens_db.chickens[elder_item_chicken.indexer*2].chicken_name
                horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                font.pointSize: 10
                color: "black"
            }

            Text {
                id: elder_chicken1_price
                x: 60
                y: 298
                width: 200
                height: 40
                text: chickens_db.chickens[elder_item_chicken.indexer*2].chicken_price
                horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                font.pointSize: 10
                color: "black"
            }

            // JavaScript 함수 정의
            function handleBagSignal(item_name) {

                if (item_name === chickens_db.chickens[elder_item_chicken.indexer*2].chicken_name) {
                    elder_chicken_img_button_1.clicked();
                }
            }


            Connections{
                target: server_signal
                onBagSignal: {
                    handleBagSignal(item_name)
                }
            }




            Button {
                id: elder_chicken_img_button_1
                x: 0
                y: 0
                width: 200
                height: 250
                opacity: 0
                onClicked: {
                    var this_index = 0
                    this_index = elder_item_chicken.indexer*2
                    var newItem = Qt.createQmlObject(
                                'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 250; height: 200; color: "lightgray"; property int c_counter: 1; property int this_index: 0; property int sub_total: chickens_db.chickens[this_index].chicken_price_int;
Image {
id: normal_chicken_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: chickens_db.chickens[this_index].chicken_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit

Component.onCompleted:{
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_chicken_name
x: 0
y: 200
width: 250
height: 33
text: chickens_db.chickens[this_index].chicken_name +" "+ chickens_db.chickens[this_index].chicken_price_int+"원"
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: chicken_count_row
x: 70
y: 250
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
c_counter-= 1
sub_total = c_counter * chickens_db.chickens[this_index].chicken_price_int
itemRow.updateParentAmount(1)
if (c_counter < 1){
c_counter = 0
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
c_counter+=1
sub_total = c_counter * chickens_db.chickens[this_index].chicken_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}',itemRow)
                    //enabled = false
                    newItem.this_index = this_index
                    if (newItem === null) {
                        console.log("Error creating object")
                    } else {
                        newItem.parent = itemRow
                    }
                }
            }
        }

        Rectangle {
            id: elder_chicken2
            x: 551
            y: 54
            width: 320
            height: 340
            color: "#ffffff"
            visible: true

            Image {
                id: elder_chicken2_img
                sourceSize.height: 260
                sourceSize.width: 320
                anchors.fill: parent
                verticalAlignment: Image.AlignTop
                source: chickens_db.chickens[elder_item_chicken.indexer*2+1].chicken_image // 이미지 경로 설정
                fillMode: Image.PreserveAspectFit

            }

            Text {
                id: elder_chicken2_name
                x: 60
                y: 260
                width: 200
                height: 33
                text: chickens_db.chickens[elder_item_chicken.indexer*2+1].chicken_name
                horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                font.pointSize: 10
                color: "black"
            }

            Text {
                id: elder_chicken2_price
                x: 60
                y: 298
                width: 200
                height: 40
                text: chickens_db.chickens[elder_item_chicken.indexer*2+1].chicken_price
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

                if (item_name === chickens_db.chickens[elder_item_chicken.indexer*2+1].chicken_name) {
                    elder_chicken_img_button_2.clicked();
                }
            }


            Button {
                id: elder_chicken_img_button_2
                x: 0
                y: 0
                width: 200
                height: 250
                opacity: 0
                onClicked: {
                    var this_index = 0
                    this_index = elder_item_chicken.indexer*2+1
                    var newItem = Qt.createQmlObject(
                                'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 250; height: 200; color: "lightgray"; property int c_counter: 1; property int this_index: 0; property int sub_total: chickens_db.chickens[this_index].chicken_price_int;
Image {
id: normal_chicken_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: chickens_db.chickens[this_index].chicken_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit

Component.onCompleted:{
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_chicken_name
x: 0
y: 200
width: 250
height: 33
text: chickens_db.chickens[this_index].chicken_name +" "+ chickens_db.chickens[this_index].chicken_price_int+"원"
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: chicken_count_row
x: 70
y: 250
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
c_counter-= 1
sub_total = c_counter * chickens_db.chickens[this_index].chicken_price_int
itemRow.updateParentAmount(1)
if (c_counter < 1){
c_counter = 0
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
c_counter+=1
sub_total = c_counter * chickens_db.chickens[this_index].chicken_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}',itemRow)
                    //enabled = false
                    newItem.this_index = this_index
                    if (newItem === null) {
                        console.log("Error creating object")
                    } else {
                        newItem.parent = itemRow
                    }
                }
            }
        }

        Button {
            id: elder_refresh_chicken
            x: 909
            y: 243
            width: 119
            height: 131
            text: qsTr("다른\n메뉴")
            font.pointSize: 10
            font.family: "Pretendard"
            onClicked: {
                if(elder_item_chicken.indexer === 1){
                    elder_item_chicken.indexer = 0
                }
                else{
                    elder_item_chicken.indexer = 1
                }
            }
        }
    }

    Item {
        id: elder_item_beverage
        x: 0
        y: 948
        width: 1080
        height: 400

        property int indexer : 0

        Label {
            id: elder_label_bevarage
            x: 74
            y: 21
            width: 36.275
            height: 28
            text: qsTr("음료")
            font.pointSize: 16
            font.family: "Pretendard"
        }

        Rectangle {
            id: elder_beverage1
            x: 147
            y: 54
            width: 320
            height: 340
            color: "#ffffff"
            visible: true

            Image {
                id: elder_beverage1_img
                anchors.fill: parent
                verticalAlignment: Image.AlignTop // 이미지가 Rectangle을 채우도록 설정
                source: beverages_db.beverages[elder_item_beverage.indexer*2].beverage_image // 이미지 경로 설정
                fillMode: Image.PreserveAspectFit // 이미지가 비율을 유지하며 Rectangle을 채우도록 설정
                sourceSize.height: 260
                sourceSize.width: 320
            }

            Text {
                id: elder_beverage1_name
                x: 60
                y: 260
                width: 200
                height: 33
                text: beverages_db.beverages[elder_item_beverage.indexer*2].beverage_name
                horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                font.pointSize: 10
                color: "black"
            }

            Text {
                id: elder_beverage1_price
                x: 60
                y: 298
                width: 200
                height: 40
                text: beverages_db.beverages[elder_item_beverage.indexer*2].beverage_price
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

                if (item_name === beverages_db.beverages[elder_item_beverage.indexer*2].beverage_name) {
                    elder_beverage_img_button_1.clicked();
                }
            }


            Button {
                id: elder_beverage_img_button_1
                x: 0
                y: 0
                width: 200
                height: 250
                opacity: 0
                onClicked: {
                    var this_index = 0
                    this_index = elder_item_beverage.indexer*2
                    var newItem = Qt.createQmlObject(
                                'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 250; height: 200; color: "lightgray"; property int c_counter: 1; property int this_index: 0; property int sub_total: beverages_db.beverages[this_index].beverage_price_int;
Image {
id: normal_beverage_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: beverages_db.beverages[this_index].beverage_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit

Component.onCompleted:{
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_beverage_name
x: 0
y: 200
width: 250
height: 33
text: beverages_db.beverages[this_index].beverage_name +" "+ beverages_db.beverages[this_index].beverage_price_int+"원"
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: beverage_count_row
x: 70
y: 250
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
c_counter-= 1
sub_total = c_counter * beverages_db.beverages[this_index].beverage_price_int
itemRow.updateParentAmount(1)
if (c_counter < 1){
c_counter = 0
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
c_counter+=1
sub_total = c_counter * beverages_db.beverages[this_index].beverage_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}',itemRow)
                    //enabled = false
                    newItem.this_index = this_index
                    if (newItem === null) {
                        console.log("Error creating object")
                    } else {
                        newItem.parent = itemRow
                    }
                }
            }
        }

        Rectangle {
            id: elder_beverage2
            x: 551
            y: 52
            width: 320
            height: 340
            color: "#ffffff"
            visible: true

            Image {
                id: elder_beverage2_img
                anchors.fill: parent
                verticalAlignment: Image.AlignTop // 이미지가 Rectangle을 채우도록 설정
                source: beverages_db.beverages[elder_item_beverage.indexer*2+1].beverage_image // 이미지 경로 설정
                fillMode: Image.PreserveAspectFit // 이미지가 비율을 유지하며 Rectangle을 채우도록 설정
                sourceSize.height: 260
                sourceSize.width: 320
            }

            Text {
                id: elder_beverage2_name
                x: 60
                y: 260
                width: 200
                height: 33
                text: beverages_db.beverages[elder_item_beverage.indexer*2+1].beverage_name
                horizontalAlignment: Text.AlignHCenter // 텍스트를 Rectangle의 중앙에 배치
                font.pointSize: 10
                color: "black"
            }

            Text {
                id: elder_beverage2_price
                x: 60
                y: 298
                width: 200
                height: 40
                text: beverages_db.beverages[elder_item_beverage.indexer*2+1].beverage_price
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

                if (item_name === beverages_db.beverages[elder_item_beverage.indexer*2+1].beverage_namee) {
                    elder_beverage_img_button_2.clicked();
                }
            }


            Button {
                id: elder_beverage_img_button_2
                x: 0
                y: 0
                width: 200
                height: 250
                opacity: 0
                onClicked: {
                    var this_index = 0
                    this_index = elder_item_beverage.indexer*2+1
                    var newItem = Qt.createQmlObject(
                                'import QtQuick 6.2; import QtQuick.Controls 6.2;
Rectangle { width: 250; height: 200; color: "lightgray"; property int c_counter: 1; property int this_index: 0; property int sub_total: beverages_db.beverages[this_index].beverage_price_int;
Image {
id: normal_beverage_img
anchors.fill: parent
verticalAlignment: Image.AlignTop
source: beverages_db.beverages[this_index].beverage_image
sourceSize.height: 150
sourceSize.width: 200
fillMode: Image.PreserveAspectFit

Component.onCompleted:{
itemRow.updateParentAmount(1)
}
}

Text {
id: normal_beverage_name
x: 0
y: 200
width: 250
height: 33
text: beverages_db.beverages[this_index].beverage_name +" "+ beverages_db.beverages[this_index].beverage_price_int+"원"
verticalAlignment: Text.AlignVCenter
horizontalAlignment: Text.AlignHCenter
font.pointSize: 10
color: "black"
}

Row {
id: beverage_count_row
x: 70
y: 250
spacing: 10
anchors.horizontalCenter: parent.horizontalCenter


Button {
text: "-"
onClicked: {
c_counter-= 1
sub_total = c_counter * beverages_db.beverages[this_index].beverage_price_int
itemRow.updateParentAmount(1)
if (c_counter < 1){
c_counter = 0
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
c_counter+=1
sub_total = c_counter * beverages_db.beverages[this_index].beverage_price_int
itemRow.updateParentAmount(1)
}
font.pointSize: 10
}
}
}',itemRow)
                    //enabled = false
                    newItem.this_index = this_index
                    if (newItem === null) {
                        console.log("Error creating object")
                    } else {
                        newItem.parent = itemRow
                    }
                }
            }
        }

        Button {
            id: elder_refresh_bev
            x: 909
            y: 243
            width: 119
            height: 131
            text: qsTr("다른\n메뉴")
            font.pointSize: 10
            font.family: "Pretendard"
            onClicked: {
                if(elder_item_beverage.indexer === 1){
                    elder_item_beverage.indexer = 0
                }
                else{
                    elder_item_beverage.indexer = 1
                }
            }
        }
    }

    Rectangle {
        id: elder_buttons
        x: 0
        y: 1779
        width: 1080
        height: 141
        color: "#f5efb3"
        border.width: 0

        Button {
            id: elder_button_pay
            x: 0
            y: 0
            width: 1080
            height: 141
            text: qsTr("결제")
            font.pointSize: 30
            font.family: "Pretendard"
            Material.background: Material.Amber
            onClicked: elder_modalPopup.visible = true
        }
    }

    Button {
        id: elder_button_pop
        x: 138
        y: 40
        width: 338
        height: 124
        text: qsTr("초기 화면으로")
        Material.background: Material.Amber
        font.bold: true
        font.pointSize: 18
        font.family: "Pretendard"

        Connections {
            target: elder_button_pop
            onClicked: {
                stackView.pop()
            }
        }
    }

    Button {
        id: elder_button_voice
        x: 539
        y: 40
        width: 338
        height: 124
        text: qsTr("음성 서비스")
        Material.background: Material.Amber
        font.pointSize: 18
        font.family: "Pretendard"
        font.bold: true
        onClicked: {
            stackView.pop()
            stackView.push(Qt.resolvedUrl(("Screen02_01.qml")))
            server_signal.setVS(true)
        }
        Connections {
            target: elder_button_voice
        }
    }

    Popup {
        id: elder_modalPopup
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
                        id: elder_pay_yes
                        text: "네"
                        font.pointSize: 10
                        font.family: "Pretendard"
                        onClicked: elder_modalPopup.visible = false

                        Connections {
                            target: elder_pay_yes
                            onClicked: {
                                stackView.push(Qt.resolvedUrl(
                                                   ("Screen03.qml")))
                            }
                        }
                    }
                    Button {
                        id: elder_pay_no
                        text: "아니오"
                        font.pointSize: 10
                        font.family: "Pretendard"
                        onClicked: elder_modalPopup.visible = false
                    }
                }
            }
        }
    }
}
