#include "burgers.hpp"
#include <QString>
#include <vector>
#include <string>

Burgers::Burgers(QObject *parent, int id, QString name, QString image, QString price, int priceInt)
    : QObject(parent), m_burger_id(id), m_burger_name(name), m_burger_image(image), m_burger_price(price), m_burger_price_int(priceInt)
{

}

Burgers::~Burgers(){ // 소멸자 정의
    // 벡터에 저장된 Burgers 객체들을 삭제
    for (auto burger : m_burgers) {
        delete burger;
    }
}
void Burgers::burgerSlot(){

    emit burgerInitialized();
}
int Burgers::getBurgerId(){
    return m_burger_id;
}
QString Burgers::getBurgerName(){
    return m_burger_name;
}
QString Burgers::getBurgerImage(){
    return m_burger_image;
}
QString Burgers::getBurgerPrice(){
    return m_burger_price;
}
int Burgers::getburgerPriceInt(){
    return m_burger_price_int;
}
std::vector<Burgers*> Burgers::getBurgers(){
    return m_burgers;
}

void Burgers::init(){
    setBurgers();
}

void Burgers::setBurgers(){
    // 객체 간의 관계를 표현하기 위해 std::vector 사용
    std::vector<std::string> burger_names = {"불고기 버거","베이컨 버거","치즈 버거","치킨 버거"};
    std::vector<std::string> burger_prices = {"3000원","3500원","2500원","2000원"};
    std::vector<int> burger_pricesInt = {3000,3500,2500,2000};

    for (int i = 0; i < 4; i++) {
        // 새로운 Burgers 객체를 동적으로 생성하여 벡터에 추가
        QString b_name = QString::fromStdString(burger_names[i]);
        QString b_image = QString::fromStdString("images/burger" + std::to_string(i) +".jpg");
        QString b_price = QString::fromStdString(burger_prices[i]);
        m_burgers.push_back(new Burgers(nullptr, i,b_name,b_image,b_price,burger_pricesInt[i]));
    }
}
