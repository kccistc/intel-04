#include "voiceQnA.h"
#include <QObject>
#include <QString>
#include <vector>
#include <iostream>

// 정적 변수 초기화
VoiceQnA* VoiceQnA::instance = nullptr;

// 싱글톤 인스턴스를 얻는 메소드
VoiceQnA* VoiceQnA::getInstance(QObject *parent) {
    if (instance == nullptr) {
        instance = new VoiceQnA(parent);
    }
    return instance;
}

// 생성자
VoiceQnA::VoiceQnA(QObject *parent, QString voice_question, QString voice_answer)
    : QObject(parent), m_voice_question(voice_question), m_voice_answer(voice_answer)
{
}

// 소멸자
VoiceQnA::~VoiceQnA()
{
}

// 음성 질문을 가져오는 메소드
QString VoiceQnA::getQuestion() const
{
    return m_voice_question;
}

// 음성 답변을 가져오는 메소드
QString VoiceQnA::getAnswer() const
{
    return m_voice_answer;
}

// 모든 QnA 인스턴스를 가져오는 메소드
std::vector<VoiceQnA*> VoiceQnA::getQnAs() const
{
    return m_QnAs;
}

// 새로운 QnA 인스턴스를 추가하는 메소드
void VoiceQnA::AppendQnAs(const QString &question, const QString &answer)
{
    VoiceQnA* new_qna = new VoiceQnA(nullptr, question, answer);
    m_QnAs.push_back(new_qna);
    setQuestion(question);
    setAnswer(answer);
    emit newQnAAdded(question, answer); // 시그널 발생
}

// 음성 질문을 설정하는 메소드
void VoiceQnA::setQuestion(const QString &question)
{
    if (m_voice_question != question) {
        m_voice_question = question;
        emit voiceQuestionChanged();
    }
}

// 음성 답변을 설정하는 메소드
void VoiceQnA::setAnswer(const QString &answer)
{
    if (m_voice_answer != answer) {
        m_voice_answer = answer;
        emit voiceAnswerChanged();
    }
}

// QnA 인스턴스를 초기화하는 메소드
void VoiceQnA::init()
{
    emit VoiceQnAInitialized();
}

// 슬롯 구현
void VoiceQnA::VoiceQnASlot()
{
    // 슬롯 처리 코드
    std::cout << "VoiceQnASlot called" << std::endl;
}
