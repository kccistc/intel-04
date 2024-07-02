#ifndef SERVER_H
#define SERVER_H

#include <QObject>
#include <QTcpServer>
#include <QTcpSocket>
#include <QJsonObject>
#include <QProcess>
#include <QVariant>
#include "worker.h"


class Server : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool blind_wheelchair READ getBW WRITE setBW NOTIFY bwChanged)
    Q_PROPERTY(bool start_camera READ getSC WRITE setSC NOTIFY scChanged)
    Q_PROPERTY(bool voice_state READ getVS WRITE setVS NOTIFY vsChanged)

public:
    explicit Server(QObject *parent = nullptr);
    void pathInit();
    void setPath(std::string path);
    std::string getPath();
    void emitWithDelay(const QVariant &value, int delayMs);

public slots:
    void startServer(const QString &ipAddress, quint16 port);
    void stopServer();
    void sendMessage(const QString &message);
    bool getBW();
    void setBW(bool data);
    bool getSC();
    void setSC(bool data);
    bool getVS();
    void setVS(bool data);
    void turnOnVoiceService();
    void turnOnCamera();
    void handleNewQnA(QString question, QString answer); // 새로운 슬롯 추가
    void sendToggleServo();

signals:
    void bwChanged();
    void vsChanged();
    void scChanged();
    void voiceServiceFinished();
    void cameraServiceFinished();
    void bagSignal(QString item_name);
    void choiceSignal(QString method_name);
    void completeSignal();
    void toggleServo();




private slots:
    void onNewConnection();
    void onDisconnected();
    void onReadyRead();
    void onSocketStateChanged(QAbstractSocket::SocketState socketState);
    void onError(QAbstractSocket::SocketError socketError);
    void onVoiceWorkFinished();
    void onCameraWorkFinished();


private:
    QTcpServer *m_server;
    QList<QTcpSocket *> m_clients;
    bool m_blind_wheelchair = false;
    bool m_start_camera = false;
    bool m_voice_state = false;
    std::string m_path = "";

    QThread *m_voiceWorkerProcess;
    QThread *m_cameraWorkerProcess;
};

#endif // SERVER_H
