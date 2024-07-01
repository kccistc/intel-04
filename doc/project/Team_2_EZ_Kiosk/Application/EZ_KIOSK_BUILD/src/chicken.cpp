#include "chicken.hpp"
#include <QString>
#include <vector>
#include <string>

Chickens::Chickens(QObject *parent, int id, QString name, QString image, QString price, int priceInt)
    : QObject(parent), m_chicken_id(id), m_chicken_name(name), m_chicken_image(image), m_chicken_price(price), m_chicken_price_int(priceInt)
{

}

Chickens::~Chickens(){ // 소멸자 정의
    // 벡터에 저장된 Chickens 객체들을 삭제
    for (auto Chicken : m_chickens) {
        delete Chicken;
    }
}
void Chickens::chickenSlot(){
    setChickens();
    emit chickenInitialized();
}
int Chickens::getchickenId(){
    return m_chicken_id;
}
QString Chickens::getchickenName(){
    return m_chicken_name;
}
QString Chickens::getchickenImage(){
    return m_chicken_image;
}
QString Chickens::getchickenPrice(){
    return m_chicken_price;
}
int Chickens::getchickenPriceInt(){
    return m_chicken_price_int;
}
std::vector<Chickens*> Chickens::getChickens(){
    return m_chickens;
}

void Chickens::init(){
    setChickens();
}

void Chickens::setChickens(){
    // 객체 간의 관계를 표현하기 위해 std::vector 사용

    std::vector<std::string> chicken_names = {"후라이드","양념 치킨","간장 치킨","치킨 너겟"};
    std::vector<std::string> chicken_prices = {"8000원","8500원","8500원","5000원"};
    std::vector<int> chicken_pricesInt = {8000,8500,8500,5000};

    for (int i = 0; i < 4; i++) {
        // 새로운 Chickens 객체를 동적으로 생성하여 벡터에 추가
        QString c_name = QString::fromStdString(chicken_names[i]);
        QString c_image = QString::fromStdString("images/chicken" + std::to_string(i) +".jpeg");
        QString c_price = QString::fromStdString(chicken_prices[i]);
        m_chickens.push_back(new Chickens(nullptr, i,c_name,c_image,c_price,chicken_pricesInt[i]));
    }
}
