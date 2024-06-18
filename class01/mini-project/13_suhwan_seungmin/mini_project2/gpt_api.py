import openai

OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY

def summarize_and_save_text(input_file, output_file='summary.txt'):
    # input_file에서 텍스트 읽기
    with open(input_file, 'r', encoding='utf-8') as file:
        query = file.read()

    # 모델 - GPT 3.5 Turbo 선택
    model = "gpt-3.5-turbo"

    # 메시지 설정하기
    messages = [{
        "role": "system",
        "content": "You are the best linguist in the world who can summarize texts easily to understand. The given text should be summarized briefly."
    }, {
        "role": "user",
        "content": query
    }]

    # ChatGPT API 호출하기
    response = openai.ChatCompletion.create(model=model, messages=messages)
    answer = response['choices'][0]['message']['content']
    #print(f"요약본이 {output_file} 파일에 저장되었습니다.")

    return answer


