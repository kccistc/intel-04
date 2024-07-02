// socketthread.cpp
#include "socketthread.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>

SocketThread::SocketThread(quint16 port, QObject *parent)
    : QThread(parent), server(nullptr), port(port)
{
}

SocketThread::~SocketThread()
{
    if (server) {
        server->close();
        server->deleteLater();
    }
    for (QTcpSocket *client : clients) {
        client->disconnectFromHost();
        client->deleteLater();
    }
}

void SocketThread::run()
{
    server = new QTcpServer();

    connect(server, &QTcpServer::newConnection, this, &SocketThread::onNewConnection);

    if (!server->listen(QHostAddress::LocalHost, port)) {
        qDebug() << "Server could not start: " << server->errorString();
        return;
    } else {
        qDebug() << "Server started on port" << port;
    }

    exec();
}

void SocketThread::onNewConnection()
{
    QTcpSocket *clientSocket = server->nextPendingConnection();

    connect(clientSocket, &QTcpSocket::readyRead, this, &SocketThread::onReadyRead);
    connect(clientSocket, &QTcpSocket::disconnected, this, &SocketThread::onClientDisconnected);
    clients.append(clientSocket);

    qDebug() << "New client connected from" << clientSocket->peerAddress().toString();
}

void SocketThread::onReadyRead()
{
    QTcpSocket *clientSocket = qobject_cast<QTcpSocket*>(sender());

    while (clientSocket->canReadLine()) {
        QByteArray data = clientSocket->readAll();  // readAll로 변경하여 모든 데이터를 읽음
        qDebug() << "Data from client:" << data;
        QJsonDocument doc = QJsonDocument::fromJson(data);

        if (!doc.isNull() && doc.isObject()) {
            QJsonObject json = doc.object();
            qDebug() << "Received JSON:";
            qDebug() << "Message Type:" << json["message_type"].toString();
            qDebug() << "Value:" << json["value"].toString();
            // 클라이언트로부터 받은 데이터를 터미널에 출력
            qDebug() << "Data from client:" << data;
            emit dataReceived(data);

            // 응답 전송
            QJsonObject response;
            response["status"] = "received";
            QJsonDocument responseDoc(response);
            //clientSocket->SocketThread::onSocketWrite(responseDoc.toJson(QJsonDocument::Compact));
            clientSocket->write(responseDoc.toJson(QJsonDocument::Compact));
            clientSocket->flush();
        }
    }
}

void SocketThread::onClientDisconnected()
{
    QTcpSocket *clientSocket = qobject_cast<QTcpSocket*>(sender());
    if (!clientSocket)
        return;

    clients.removeAll(clientSocket);
    clientSocket->deleteLater();

    qDebug() << "Client disconnected from" << clientSocket->peerAddress().toString();
}
