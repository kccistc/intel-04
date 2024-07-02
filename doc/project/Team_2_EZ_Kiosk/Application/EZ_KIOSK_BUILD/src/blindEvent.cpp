#include "blindEvent.hpp"
#include <QObject>
#include <QDebug>

BlindEvent::BlindEvent(QObject *parent)
    : QObject(parent), m_bflag(0)
{

}

BlindEvent::~BlindEvent() // 소멸자 정의
{
    m_bflag = 0;
    emit bflagChanged();
}


void BlindEvent::blindSlot(const QByteArray &data)
{
    qDebug() << "Data received in main thread:";
    qDebug() << data;
}

int BlindEvent::getbFlag(){
    return m_bflag;
}

void BlindEvent::setbFlag(int digit){
    m_bflag = digit;
    emit bflagChanged();
}

