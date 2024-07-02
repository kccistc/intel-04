// Copyright (C) 2021 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR GPL-3.0-only

import QtQuick 6.2
import EZ_KIOSK_CLEAN
import QtQuick.VirtualKeyboard 6.2
import QtQuick.Controls 6.2

Window {
    width: 1070
    height: 1900

    visible: true
    title: "EZ_KIOSK_CLEAN"

    StackView{
        id : stackView
        initialItem: mainScreen
        anchors.fill: parent
    }

    Screen00 {
        id: mainScreen
    }
}

