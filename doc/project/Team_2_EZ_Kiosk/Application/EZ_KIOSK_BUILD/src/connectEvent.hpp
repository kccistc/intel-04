#ifndef CONNECTEVENT_HPP
#define CONNECTEVENT_HPP

#include <QObject>

class ConnectEvent : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int result READ getResult WRITE setResult NOTIFY resultChanged)

public:
    explicit ConnectEvent(QObject *parent = nullptr);
    ~ConnectEvent();
    int getResult();
    void setResult(int digit);


signals:
    void resultChanged(); // 신호 선언

public slots:
    void cppSlot(); // 슬롯 선언

private:
    int m_result;
};
 
#endif // CONNECTEVENT_HPP
