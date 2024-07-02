#include "beverage.hpp"
#include <QString>
#include <vector>
#include <string>

Beverages::Beverages(QObject *parent, int id, QString name, QString image, QString price, int priceInt)
    : QObject(parent), m_beverage_id(id), m_beverage_name(name), m_beverage_image(image), m_beverage_price(price), m_beverage_price_int(priceInt)
{

}

Beverages::~Beverages(){ // 소멸자 정의
    // 벡터에 저장된 Beverages 객체들을 삭제
    for (auto beverage : m_beverages) {
        delete beverage;
    }
}
void Beverages::beverageSlot(){
    setBeverages();
    emit beverageInitialized();
}
int Beverages::getbeverageId(){
    return m_beverage_id;
}
QString Beverages::getbeverageName(){
    return m_beverage_name;
}
QString Beverages::getbeverageImage(){
    return m_beverage_image;
}
QString Beverages::getbeveragePrice(){
    return m_beverage_price;
}
int Beverages::getbeveragePriceInt(){
    return m_beverage_price_int;
}
std::vector<Beverages*> Beverages::getbeverages(){
    return m_beverages;
}

void Beverages::init(){
    setBeverages();
}


void Beverages::setBeverages(){
    // 객체 간의 관계를 표현하기 위해 std::vector 사용

    std::vector<std::string> beverage_names = {"콜라","사이다","아이스티","모히또"};
    std::vector<std::string> beverage_prices = {"1500원","1500원","1000원","2000원"};
    std::vector<int> beverage_pricesInt = {1500,1500,1000,2000};

    for (int i = 0; i < 4; i++) {
        // 새로운 Beverages 객체를 동적으로 생성하여 벡터에 추가
        QString be_name = QString::fromStdString(beverage_names[i]);
        QString be_image = QString::fromStdString("images/beverage" + std::to_string(i) +".jpeg");
        QString be_price = QString::fromStdString(beverage_prices[i]);
        m_beverages.push_back(new Beverages(nullptr, i,be_name,be_image,be_price,beverage_pricesInt[i]));
    }
}
