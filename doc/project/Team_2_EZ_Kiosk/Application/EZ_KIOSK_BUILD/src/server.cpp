#include "server.h"
#include "worker.h"
#include "voiceQnA.h"
#include <QDebug>
#include <QOverload>
#include <QJsonDocument>
#include <QJsonObject>
#include <filesystem>
#include <QThread>
#include <QHostAddress>
#include <QTimer>
#include <QVariant>

Server::Server(QObject *parent)
    : QObject(parent), m_server(new QTcpServer(this)), m_voiceWorkerProcess(nullptr), m_cameraWorkerProcess(nullptr)
{
    connect(m_server, &QTcpServer::newConnection, this, &Server::onNewConnection);
    //connect(this, &Server::bwChanged, this, &Server::sendToggleServo); // bwChanged 시그널을 turnOnVoiceService 슬롯에 연결
    connect(this, &Server::scChanged, this, &Server::turnOnCamera);
    connect(this, &Server::vsChanged, this, &Server::turnOnVoiceService);
    connect(VoiceQnA::getInstance(), &VoiceQnA::newQnAAdded, this, &Server::handleNewQnA);
    pathInit();
}

void Server::startServer(const QString &ipAddress,quint16 port)
{
    QHostAddress address(ipAddress);
    if (m_server->listen(address, port)) {
        qDebug() << "Server is listening on IP" << ipAddress << "and port" << port;
    } else {
        qDebug() << "Server could not start. Reason:" << m_server->errorString();
    }

    qDebug() << "Server is listening on port" << port;
}

void Server::stopServer()
{
    m_server->close();
    for (QTcpSocket *client : m_clients) {
        client->disconnectFromHost();
        client->deleteLater();
    }
}

void Server::sendMessage(const QString &message)
{
    QByteArray data = message.toUtf8();
    for (QTcpSocket *client : m_clients) {
        client->write(data);
        client->flush();
    }
}

void Server::onNewConnection()
{
    QTcpSocket *clientSocket = m_server->nextPendingConnection();
    connect(clientSocket, &QTcpSocket::disconnected, this, &Server::onDisconnected);
    connect(clientSocket, &QTcpSocket::readyRead, this, &Server::onReadyRead);
    connect(clientSocket, &QTcpSocket::stateChanged, this, &Server::onSocketStateChanged);

    // 오류 시그널을 QOverload를 사용하여 SocketError에 대한 슬롯과 연결합니다.
    connect(clientSocket, &QTcpSocket::errorOccurred, this, &Server::onError);

    m_clients.append(clientSocket);
    qDebug() << "새 클라이언트가" << clientSocket->peerAddress().toString() << "에서 접속하였습니다.";
}

void Server::onDisconnected()
{
    QTcpSocket *clientSocket = qobject_cast<QTcpSocket *>(sender());
    if (!clientSocket)
        return;

    qDebug() << "Client disconnected from" << clientSocket->peerAddress().toString();
    m_clients.removeAll(clientSocket);
    clientSocket->deleteLater();
}

void Server::onReadyRead()
{
    QTcpSocket *clientSocket = qobject_cast<QTcpSocket *>(sender());
    if (!clientSocket)
        return;

    QByteArray jsonData = clientSocket->readAll();
    QJsonParseError error;

    QJsonDocument doc = QJsonDocument::fromJson(jsonData, &error);

    if (error.error != QJsonParseError::NoError) {
        qDebug() << "Error parsing JSON:" << error.errorString();
        return;
    }

    if (!doc.isNull() && doc.isObject()) {
        QJsonObject jsonObj = doc.object();

        qDebug() << jsonObj["message_type"];

        if(jsonObj["message_type"] == "client_classification"){
            if(jsonObj["value"] == "Wheelchair"){
                sendToggleServo();
                setBW(true);
            }else if(jsonObj["value"] == "Blind"){
                setBW(true);
            }
        }else if (jsonObj["message_type"] == "ultrasonic wave sensor"){
            if(jsonObj["value"] == "True"){
                setSC(true);
            }
        }else if (jsonObj["message_type"] == "voice_qna") {
            QString question = jsonObj["question"].toString();
            QString answer = jsonObj["answer"].toString();

            // 싱글톤 인스턴스를 가져와서 새로운 QnA 인스턴스를 추가
            VoiceQnA::getInstance()->AppendQnAs(question, answer);
        }
        qDebug() << "Data from client:" << jsonObj;

        //if jsonObj.
        //emit dataReceived(jsonObj); // Emit the dataReceived signal with the parsed JSON object
    }

}

void Server::onSocketStateChanged(QAbstractSocket::SocketState socketState)
{
    QTcpSocket *clientSocket = qobject_cast<QTcpSocket *>(sender());
    if (!clientSocket)
        return;

    if (socketState == QAbstractSocket::UnconnectedState) {
        qDebug() << "Client socket state changed to UnconnectedState";
        m_clients.removeAll(clientSocket);
        clientSocket->deleteLater();
    }
}

void Server::onError(QAbstractSocket::SocketError socketError)
{
    QTcpSocket *clientSocket = qobject_cast<QTcpSocket *>(sender());
    if (!clientSocket)
        return;

    qDebug() << "Socket error:" << socketError << "-" << clientSocket->errorString();
}

void Server::turnOnVoiceService()
{
    if(m_voice_state == true){
        QString scriptPath = QString::fromStdString(this->getPath() + "/stt_tts_module.py");
        Worker *worker = new Worker(scriptPath);
        connect(worker, &Worker::taskFinished, worker, &QObject::deleteLater);
        connect(worker, &Worker::taskFinished, this, &Server::onVoiceWorkFinished);

        worker->turnOnVoiceService(scriptPath);
    }
}


void Server::turnOnCamera()
{
    std::string p_command = "python3 "+ this->getPath()+"/video_processing.py";
    std::system(p_command.c_str());
}

void Server::sendToggleServo()
{
    QTcpSocket socket;
    QString serverIp = "10.10.15.129";
    quint16 serverPort = 8889;

    qDebug() << "서보모터를 제어 합니다";

    socket.connectToHost(QHostAddress(serverIp), serverPort);

    // 연결이 완료될 때까지 대기
    if (socket.waitForConnected(3000)) {

        if(m_blind_wheelchair){
            QString message = "{\"message_type\": \"servo_motor\", \"value\": \"55\"}";
            QByteArray data = message.toUtf8();
            socket.write(data);
        }else{
            QString message = "{\"message_type\": \"servo_motor\", \"value\": \"145\"}";
            QByteArray data = message.toUtf8();
            socket.write(data);
        }
        // 데이터가 실제로 전송될 때까지 대기
        if (socket.waitForBytesWritten(3000)) {
            qDebug() << "Message sent to server!";
        } else {
            qDebug() << "Failed to send message!";
        }
    }


}

bool Server::getBW(){

    return m_blind_wheelchair;
}

void Server::setBW(bool data){
    m_blind_wheelchair = data;
    emit bwChanged();
}

bool Server::getSC(){

    return m_start_camera;
}

void Server::setSC(bool data){
    m_start_camera = data;
    emit scChanged();
}

bool Server::getVS(){

    return m_voice_state;
}

void Server::setVS(bool data){
    m_voice_state = data;
    emit vsChanged();
}

void Server::pathInit(){
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::string currentPathString = currentPath.string();
    setPath(currentPathString);
}

void Server::setPath(std::string path){
    this->m_path = path;
}
std::string Server::getPath(){
    return this->m_path;
}

void Server::onVoiceWorkFinished()
{
    emit voiceServiceFinished();
}

void Server::onCameraWorkFinished()
{
    emit cameraServiceFinished();
}


void Server::emitWithDelay(const QVariant &value, int delayMs)
{
    QTimer::singleShot(delayMs, this, [this, value]() {
        if (value.canConvert<QString>()) {
            QString stringValue = value.toString();
            emit choiceSignal(stringValue);
        }
    });
}
void Server::handleNewQnA(QString question, QString answer) {

    QMap<QString, QString> bagMap;
    bagMap.insert("치즈버거", "치즈 버거");
    bagMap.insert("불고기버거", "불고기 버거");
    bagMap.insert("베이컨버거", "베이컨 버거");
    bagMap.insert("치킨버거", "치킨 버거");
    bagMap.insert("후라이드", "후라이드");
    bagMap.insert("양념치킨", "양념 치킨");
    bagMap.insert("간장치킨", "간장 치킨");
    bagMap.insert("치킨너겟", "치킨 너겟");
    bagMap.insert("콜라", "콜라");
    bagMap.insert("사이다", "사이다");
    bagMap.insert("아이스티", "아이스티");
    bagMap.insert("모히또", "모히또");

    QMap<QString, QString> choiceMap;
    choiceMap.insert("카드", "card");
    choiceMap.insert("현금", "cash");
    choiceMap.insert("먹고가기", "for_here");
    choiceMap.insert("포장", "to_go");
    choiceMap.insert("만족", "good");
    choiceMap.insert("별로", "bad");

    if (answer.contains("장바구니")) {
        for (auto it = bagMap.constBegin(); it != bagMap.constEnd(); ++it) {
            if (answer.contains(it.key())) {
                emit bagSignal(it.value());
                break; // 항목을 찾으면 루프를 종료합니다.
            }
        }
    }else if (answer.contains("선택")) {
        for (auto it = choiceMap.constBegin(); it != choiceMap.constEnd(); ++it) {
            if (answer.contains(it.key())) {
                emit bagSignal(it.value());
                break; // 항목을 찾으면 루프를 종료합니다.
            }
        }
    }else if(answer.contains("결제") && answer.contains("화면")){
        emit completeSignal();
        }

}
