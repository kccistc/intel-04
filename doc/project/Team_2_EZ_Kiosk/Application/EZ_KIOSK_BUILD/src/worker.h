#ifndef WORKER_H
#define WORKER_H

#include <QThread>
#include <QProcess>

class Worker : public QThread
{
    Q_OBJECT

public:
    explicit Worker(const QString &scriptPath, QObject *parent = nullptr);
    void turnOnVoiceService(const QString &scriptPath);
    void run() override;
    void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);

signals:
    void taskFinished();

private:
    QString m_scriptPath;
    QProcess *m_process;

private slots:



};

#endif // WORKER_H
