#include <QObject>
#include <QThread>
#include <QString>
#include <QProcess>

class Worker : public QObject {
    Q_OBJECT
public:
    explicit Worker(const QString &scriptPath, QObject *parent = nullptr)
        : QObject(parent), m_scriptPath(scriptPath) {}

signals:
    void workFinished();

public slots:
    void doWork() {
        QProcess process;
        process.start("python3", QStringList() << m_scriptPath);
        process.waitForFinished();
        emit workFinished();
    }

private:
    QString m_scriptPath;
};
