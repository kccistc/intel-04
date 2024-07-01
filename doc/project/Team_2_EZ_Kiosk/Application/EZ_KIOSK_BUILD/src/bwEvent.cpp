#include "bwEvent.hpp"
#include <QObject>
#include <QDebug>
#include <QJsonObject>

BwEvent::BwEvent(QObject *parent)
    : QObject(parent), m_bwflag(0)
{

}

BwEvent::~BwEvent() // 소멸자 정의
{
    m_bwflag = 0;
    emit bwflagChanged();
}


void BwEvent::bwSlot(QJsonObject jsonObj)
{
    qDebug() << "Data received in main thread:";
    qDebug() << jsonObj;
}

int BwEvent::getbwFlag(){
    return m_bwflag;
}

void BwEvent::setbwFlag(int digit){
    m_bwflag = digit;
    emit bwflagChanged();
}

