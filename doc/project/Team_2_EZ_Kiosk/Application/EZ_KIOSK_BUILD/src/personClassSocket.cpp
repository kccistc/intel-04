#include "personClassSocket.h"

PersonClassSocket::PersonClassSocket(const QByteArray *data, quint16 port, QObject *parent)
    : QThread(parent), m_pServer(new QTcpServer(this)), m_nPort(port), m_data(data)
{
    m_bRunning = false;
}

PersonClassSocket::~PersonClassSocket()
{
    delete m_pServer;
}

bool PersonClassSocket::startServer(int port)
{
    if (!m_pServer->listen(QHostAddress::Any, port)) {
        qDebug() << "PersonClassSocket cannot open port" << port << ":" << m_pServer->errorString();
        return false;
    }

    m_nPort = port;
    connect(m_pServer, &QTcpServer::newConnection, this, &PersonClassSocket::onNewConnection);
    m_bRunning = true;
    start(); // QThread의 start 함수 호출

    qDebug() << "Server started on port" << port;
    return true;
}

void PersonClassSocket::stopServer()
{
    m_pServer->close();
    m_bRunning = false;
    wait(); // QThread의 wait 함수 호출
}

void PersonClassSocket::onNewConnection()
{
    QTcpSocket *clientSocket = m_pServer->nextPendingConnection();
    connect(clientSocket, &QTcpSocket::readyRead, this, &PersonClassSocket::onReadyRead);
    connect(clientSocket, &QTcpSocket::stateChanged, this, &PersonClassSocket::onSocketStateChanged);

    m_clientSocketMutex.lock();
    m_clientSockets.append(clientSocket);
    m_clientSocketMutex.unlock();

    qDebug() << "New client connected from" << clientSocket->peerAddress().toString();
}

void PersonClassSocket::onSocketStateChanged(QAbstractSocket::SocketState socketState)
{
    QTcpSocket *senderSocket = qobject_cast<QTcpSocket*>(sender());
    if (!senderSocket)
        return;

    if (socketState == QAbstractSocket::UnconnectedState) {
        m_clientSocketMutex.lock();
        m_clientSockets.removeOne(senderSocket);
        m_clientSocketMutex.unlock();
    }
}

void PersonClassSocket::onReadyRead()
{
    // 현재는 사용되지 않는 메서드입니다.
}

void PersonClassSocket::run()
{
    while (m_bRunning) {
        // 여기서 데이터를 보낼 준비가 되었음을 가정합니다.
        // m_data를 사용하여 적절한 데이터 처리 로직을 구현해야 합니다.
        // 예시로는 다음과 같이 데이터를 생성하여 보낼 수 있습니다.
        QByteArray dataToSend = *m_data;

        m_clientSocketMutex.lock();
        for (QTcpSocket *socket : std::as_const(m_clientSockets)) {
            emit socketWriteEvent(socket, dataToSend); // 데이터를 보내는 시그널 발생
        }
        m_clientSocketMutex.unlock();

        // 데이터가 전송된 후 적절한 대기 로직을 추가해야 할 수 있습니다.
        usleep(10); // 예시: 잠시 대기
    }
}

void PersonClassSocket::onSocketWrite(QTcpSocket *sock, QByteArray &data)
{
    int res = sock->write(data); // 데이터 전송
    if (res != data.size()) {
        qDebug() << QString("PersonClassSocket(%1) cannot send to client: sent %2 bytes out of %3").arg(m_nPort).arg(res).arg(data.size());
    }
    // 데이터 전송 후 필요한 처리를 추가할 수 있습니다.
}
