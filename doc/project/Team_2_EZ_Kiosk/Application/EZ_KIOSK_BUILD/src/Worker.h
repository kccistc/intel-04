#ifndef WORKER_H
#define WORKER_H

#include <QObject>
#include <QString>

class Worker : public QObject {
    Q_OBJECT
public:
    explicit Worker(const QString &scriptPath, QObject *parent = nullptr);

signals:
    void workFinished();

public slots:
    void doWork();

private:
    QString m_scriptPath;
};

#endif // WORKER_H
