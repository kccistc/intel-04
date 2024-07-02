#ifndef VOICEQNA_H
#define VOICEQNA_H

#include <QObject>
#include <QString>
#include <vector>

class VoiceQnA : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString voice_question READ getQuestion WRITE setQuestion NOTIFY voiceQuestionChanged)
    Q_PROPERTY(QString voice_answer READ getAnswer WRITE setAnswer NOTIFY voiceAnswerChanged)

public:
    // 싱글톤 인스턴스를 얻는 메소드
    static VoiceQnA* getInstance(QObject *parent = nullptr);

    // 기존의 메소드들
    QString getQuestion() const;
    QString getAnswer() const;
    std::vector<VoiceQnA*> getQnAs() const;
    void AppendQnAs(const QString &question, const QString &answer);
    void setQuestion(const QString &question);
    void setAnswer(const QString &answer);
    void init();

signals:
    void voiceQuestionChanged();
    void voiceAnswerChanged();
    void newQnAAdded(QString question, QString answer);
    void VoiceQnAInitialized();

public slots:
    void VoiceQnASlot();

private:
    static VoiceQnA* instance;

    explicit VoiceQnA(QObject *parent = nullptr, QString voice_question = "", QString voice_answer = "");
    ~VoiceQnA();

    QString m_voice_question;
    QString m_voice_answer;
    std::vector<VoiceQnA*> m_QnAs;
};

#endif // VOICEQNA_H
