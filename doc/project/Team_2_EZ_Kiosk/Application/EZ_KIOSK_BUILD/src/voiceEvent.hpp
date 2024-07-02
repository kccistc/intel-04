#ifndef VOICEEVENT_HPP
#define VOICEEVENT_HPP

#include <QObject>
#include <QDebug>

class VoiceEvent : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int vflag READ getvFlag WRITE setvFlag NOTIFY vflagChanged)

public:
    explicit VoiceEvent(QObject *parent = nullptr);
    ~VoiceEvent();
    int getvFlag();
    void setvFlag(int digit);


signals:
    void vflagChanged(); // 신호 선언

public slots:
    void voiceSlot(const QByteArray &data); // 슬롯 선언

private:
    int m_vflag;
};
 
#endif // VOICEEVENT_HPP
