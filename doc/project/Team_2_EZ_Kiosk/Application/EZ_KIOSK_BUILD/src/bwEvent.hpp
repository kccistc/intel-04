#ifndef BWEVENT_H
#define BWEVENT_H

#include <QObject>
#include <QDebug>

class BwEvent : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int bwflag READ getbwFlag WRITE setbwFlag NOTIFY bwflagChanged)

public:
    explicit BwEvent(QObject *parent = nullptr);
    ~BwEvent();
    int getbwFlag();
    void setbwFlag(int digit);


signals:
    void bwflagChanged(); // 신호 선언

public slots:
    void bwSlot(QJsonObject jsonObj); // 슬롯 선언

private:
    int m_bwflag;
};
 
#endif // BWEVENT_H
