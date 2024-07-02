// socketthread.h
#ifndef SOCKETTHREAD_H
#define SOCKETTHREAD_H

#include <QThread>
#include <QTcpServer>
#include <QTcpSocket>

class SocketThread : public QThread
{
    Q_OBJECT

public:
    explicit SocketThread(quint16 port, QObject *parent = nullptr);
    ~SocketThread();

signals:
    void dataReceived(const QByteArray &data); // 데이터 수신 시그널

protected:
    void run() override;

private slots:
    void onNewConnection();
    void onReadyRead();
    void onClientDisconnected();
    void onSocketWrite(QTcpSocket *sock, QByteArray &data);

private:
    QTcpServer *server;
    QList<QTcpSocket*> clients;
    quint16 port;
};

#endif // SOCKETTHREAD_H
