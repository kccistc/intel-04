#ifndef PERSONCLASSSOCKET_H
#define PERSONCLASSSOCKET_H

#include <QThread>
#include <QTcpServer>
#include <QTcpSocket>
#include <QMutex>

class PersonClassSocket : public QThread
{
    Q_OBJECT

public:
    explicit PersonClassSocket(const QByteArray *data,quint16 port,QObject *parent = nullptr);
    virtual ~PersonClassSocket();

signals:
    void socketWriteEvent(QTcpSocket *sock, QByteArray &data);

public slots:
    void onNewConnection();
    void onSocketStateChanged(QAbstractSocket::SocketState socketState);
    void onReadyRead();
    void onSocketWrite(QTcpSocket *sock, QByteArray &data);

public:
    bool startServer(int port);
    void stopServer();

protected:
    virtual void run() override;

private:
    bool m_bRunning;
    QTcpServer *m_pServer;
    int m_nPort;
    QList<QTcpSocket*> m_clientSockets;
    QMutex m_clientSocketMutex;
    QByteArray *m_data;
};

#endif // PERSONCLASSSOCKET_H
