// Copyright (C) 2021 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR GPL-3.0-only

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QObject>
#include "connectEvent.hpp"
#include "voiceEvent.hpp"
#include "burgers.hpp"
#include "chicken.hpp"
#include "beverage.hpp"
#include "app_environment.h"
#include "server.h"
#include "voiceQnA.h"
#include "import_qml_components_plugins.h"
#include "import_qml_plugins.h"

int main(int argc, char *argv[])
{
    set_qt_environment();

    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;


    ConnectEvent *connect_event = new ConnectEvent();
    VoiceEvent *voice_event = new VoiceEvent();
    Server *server_socket_local =new Server();
    Server *server_socket_RPi =new Server();
    Burgers *burgers_db = new Burgers();
    Chickens *chickens_db = new Chickens();
    Beverages *beverages_db = new Beverages();
    VoiceQnA *voice_action = nullptr;
    voice_action = VoiceQnA::getInstance();

    burgers_db->init();
    chickens_db->init();
    beverages_db->init();

    engine.rootContext()->setContextProperty("connectEvent",connect_event);
    engine.rootContext()->setContextProperty("voiceEvent",voice_event);
    engine.rootContext()->setContextProperty("burgers_db",burgers_db);
    engine.rootContext()->setContextProperty("chickens_db",chickens_db);
    engine.rootContext()->setContextProperty("beverages_db",beverages_db);
    engine.rootContext()->setContextProperty("server_signal",server_socket_local);
    engine.rootContext()->setContextProperty("RPi_signal",server_socket_RPi);
    engine.rootContext()->setContextProperty("voice_action",voice_action);

    const QUrl url(u"qrc:/qt/qml/Main/main.qml"_qs);

    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreated,
        &app,
        [url](QObject *obj, const QUrl &objUrl) {
            if (!obj && url == objUrl)
                QCoreApplication::exit(-1);
        },
        Qt::QueuedConnection);


    server_socket_local->startServer("0.0.0.0",8888);
    //server_socket_RPi->startServer("10.10.15.129",8889);

    engine.addImportPath(QCoreApplication::applicationDirPath() + "/qml");

    engine.addImportPath(":/");

    engine.load(url);

    if (engine.rootObjects().isEmpty()) {
        return -1;
    }

    return app.exec();
}
