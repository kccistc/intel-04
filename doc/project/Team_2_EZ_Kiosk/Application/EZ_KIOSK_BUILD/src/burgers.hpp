#ifndef BURGERS_HPP
#define BURGERS_HPP

#include <QObject>
#include <QString>
#include <vector>

class Burgers : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int burger_id READ getBurgerId CONSTANT)
    Q_PROPERTY(QString burger_name READ getBurgerName CONSTANT)
    Q_PROPERTY(QString burger_image READ getBurgerImage CONSTANT)
    Q_PROPERTY(QString burger_price READ getBurgerPrice CONSTANT)
    Q_PROPERTY(int burger_price_int READ getburgerPriceInt CONSTANT)
    Q_PROPERTY(std::vector<Burgers*> burgers READ getBurgers CONSTANT)

public:
    explicit Burgers(QObject *parent=nullptr, int id=0, QString name="", QString image="", QString price="", int priceInt=0);
    ~Burgers();
    int getBurgerId();
    QString getBurgerName();
    QString getBurgerImage();
    QString getBurgerPrice();
    std::vector<Burgers*> getBurgers();
    int getburgerPriceInt();
    void setBurgers();
    void init();

signals:
    void burgerInitialized(); // 신호 선언

public slots:
    void burgerSlot(); // 슬롯 선언

private:
    int m_burger_id;
    QString m_burger_name;
    QString m_burger_image;
    QString m_burger_price;
    int m_burger_price_int;
    std::vector<Burgers*> m_burgers;
};
 
#endif // BURGERS_HPP
