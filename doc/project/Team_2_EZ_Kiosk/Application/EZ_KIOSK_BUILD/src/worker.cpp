#include "worker.h"
#include <QDebug>

// 생성자
Worker::Worker(const QString &scriptPath, QObject *parent)
    : QThread(parent)
{
}

// 이 메서드는 스레드가 시작될 때 실행됩니다
void Worker::run()
{
    if (!m_scriptPath.isEmpty()) {
        qDebug() << "Executing script:" << m_scriptPath;

        // 스크립트 실행 시, 실제 코드 대신 데모로 간단한 메시지 출력
        qDebug() << "Voice service turned on.";
        system(qPrintable("python3 " + m_scriptPath));

        emit taskFinished();
    } else {
        qDebug() << "No script path provided.";
    }
}

// 이 슬롯은 스레드에서 실행됩니다
void Worker::turnOnVoiceService(const QString &scriptPath)
{
    m_scriptPath = scriptPath;
    start();
}
