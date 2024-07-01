#include "voiceEvent.hpp"

// 생성자
VoiceEvent::VoiceEvent(QObject *parent)
    : QObject(parent), m_vflag(0) // 초기값 설정
{
}

// 소멸자
VoiceEvent::~VoiceEvent()
{
}

// vFlag를 가져오는 메소드
int VoiceEvent::getvFlag()
{
    return m_vflag;
}

// vFlag를 설정하는 메소드
void VoiceEvent::setvFlag(int digit)
{
    if (m_vflag != digit) {
        m_vflag = digit;
        emit vflagChanged(); // vFlag가 변경되었음을 알리는 신호 발생
    }
}

// 슬롯 구현
void VoiceEvent::voiceSlot(const QByteArray &data)
{
    qDebug() << "Received data:" << data;
    // data를 처리하고 필요한 작업 수행
}
