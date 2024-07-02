#include "connectEvent.hpp"

ConnectEvent::ConnectEvent(QObject *parent)
    : QObject(parent), m_result(1)
{
}

ConnectEvent::~ConnectEvent() // 소멸자 정의
{
    m_result = 0;
    emit resultChanged();
}


void ConnectEvent::cppSlot()
{

}

int ConnectEvent::getResult(){
    return m_result;
}

void ConnectEvent::setResult(int digit){
    m_result = digit;
    emit resultChanged();
}
