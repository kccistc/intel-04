#ifndef BEVERAGE_H
#define BEVERAGE_H

#include <QObject>
#include <QString>
#include <vector>

class Beverages : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int beverage_id READ getbeverageId CONSTANT)
    Q_PROPERTY(QString beverage_name READ getbeverageName CONSTANT)
    Q_PROPERTY(QString beverage_image READ getbeverageImage CONSTANT)
    Q_PROPERTY(QString beverage_price READ getbeveragePrice CONSTANT)
    Q_PROPERTY(int beverage_price_int READ getbeveragePriceInt CONSTANT)
    Q_PROPERTY(std::vector<Beverages*> beverages READ getbeverages CONSTANT)

public:
    explicit Beverages(QObject *parent=nullptr, int id=0, QString name="", QString image="", QString price="", int priceInt=0);

    ~Beverages();

    int getbeverageId();
    QString getbeverageName();
    QString getbeverageImage();
    QString getbeveragePrice();
    std::vector<Beverages*> getbeverages();
    int getbeveragePriceInt();
    void setBeverages();
    void init();

signals:
    void beverageInitialized(); // 신호 선언

public slots:
    void beverageSlot(); // 슬롯 선언

private:
    int m_beverage_id;
    QString m_beverage_name;
    QString m_beverage_image;
    QString m_beverage_price;
    int m_beverage_price_int;
    std::vector<Beverages*> m_beverages;
};
 
#endif // BEVERAGE_H
