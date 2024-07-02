#ifndef BLINDEVENT_H
#define BLINDEVENT_H

#include <QObject>
#include <QDebug>

class BlindEvent : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int bflag READ getbFlag WRITE setbFlag NOTIFY bflagChanged)

public:
    explicit BlindEvent(QObject *parent = nullptr);
    ~BlindEvent();
    int getbFlag();
    void setbFlag(int digit);


signals:
    void bflagChanged(); // 신호 선언

public slots:
    //void blindSlot(const QByteArray &data); // 슬롯 선언

private:
    int m_bflag;
};
 
#endif // BLINDEVENT_H
