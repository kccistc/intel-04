#ifndef CHICKEN_H
#define CHICKEN_H

#include <QObject>
#include <QString>
#include <vector>

class Chickens : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int chicken_id READ getchickenId CONSTANT)
    Q_PROPERTY(QString chicken_name READ getchickenName CONSTANT)
    Q_PROPERTY(QString chicken_image READ getchickenImage CONSTANT)
    Q_PROPERTY(QString chicken_price READ getchickenPrice CONSTANT)
    Q_PROPERTY(int chicken_price_int READ getchickenPriceInt CONSTANT)
    Q_PROPERTY(std::vector<Chickens*> chickens READ getChickens CONSTANT)

public:
    explicit Chickens(QObject *parent=nullptr, int id=0, QString name="", QString image="", QString price="", int priceInt=0);

    ~Chickens();

    int getchickenId();
    QString getchickenName();
    QString getchickenImage();
    QString getchickenPrice();
    std::vector<Chickens*> getChickens();
    int getchickenPriceInt();
    void setChickens();
    void init();

signals:
    void chickenInitialized(); // 신호 선언

public slots:
    void chickenSlot(); // 슬롯 선언

private:
    int m_chicken_id;
    QString m_chicken_name;
    QString m_chicken_image;
    QString m_chicken_price;
    int m_chicken_price_int;
    std::vector<Chickens*> m_chickens;
};
 
#endif // CHICKEN_H
