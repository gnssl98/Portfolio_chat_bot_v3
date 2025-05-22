import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수에서 OpenAI API 키 로딩
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "info.jsonl"
OUTPUT_FILE = "refined_info.jsonl"

def regenerate_with_inference(data):
    system_prompt = (
        "You are a helpful assistant answering questions about a user's portfolio. "
        "Even if the provided context does not explicitly contain the answer, "
        "you should make a reasonable assumption based on common knowledge or typical patterns."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{data['context']}\n\nQuestion:\n{data['question']}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 또는 gpt-3.5-turbo
            messages=messages,
            temperature=0.9,  # 더 창의적인 답변 유도
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"[ERROR] GPT 응답 실패: {e}")
        return "정보가 부족합니다"

def refine_all():
    refined = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                print(f"🔄 재생성 중 ({i+1}) → {data['question']}")
                new_answer = regenerate_with_inference(data)
                data["answer"] = new_answer
                refined.append(data)
            except Exception as e:
                print(f"[ERROR] JSONL 라인 파싱 실패: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in refined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 총 {len(refined)} 개 항목 재생성 완료 → {OUTPUT_FILE}")

if __name__ == "__main__":
    refine_all()
