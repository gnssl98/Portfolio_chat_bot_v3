import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXISTING_FILE = "classification_data.jsonl"
OUTPUT_FILE = "new_classification_data.jsonl"

# 기존 질문 로딩
existing_questions = set()
if os.path.exists(EXISTING_FILE):
    with open(EXISTING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                existing_questions.add(data["text"].strip())
            except Exception:
                continue

system_prompt = "You are a helpful assistant generating JSON classification data for a Korean AI engineer portfolio chatbot."
generated_set = set()

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for label in [0, 1]:  # label 0과 1 각각 생성
        for i in range(5):  # 각 label에 대해 5회 반복
            print(f"[label={label}] [{i+1}/5] Generating new data...")

            example_text = (
                "- 당신의 전공은 무엇인가요?\n- 어떤 학교를 졸업하셨나요?\n- 어떤 프로젝트를 진행했나요?\n"
                if label == 1 else
                "- 오늘 날씨 어때요?\n- 점심 뭐 먹었어요?\n- 주말에 뭐하세요?\n"
            )

            user_prompt = f"""\
Please generate 5 new Korean sentence classification data examples as JSON lines.
- Each example must be in this format: {{"text": "...", "label": {label}}}
- Label {label} means {"portfolio-related" if label == 1 else "general daily"} question.
- Do NOT duplicate any of the following recent questions:
""" + "\n".join(f"- {q}" for q in list(existing_questions)[-50:]) + f"""\n
- Examples of label {label}:
{example_text}
Return ONLY the JSON lines.
"""

            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.8
                )

                content = response.choices[0].message.content.strip()
                lines = content.split("\n")

                for line in lines:
                    try:
                        data = json.loads(line.strip())
                        text = data["text"].strip()
                        if (
                            data.get("label") == label and
                            text not in existing_questions and
                            text not in generated_set
                        ):
                            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                            generated_set.add(text)
                    except json.JSONDecodeError:
                        continue

                time.sleep(1)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)

print(f"\n 완료: {len(generated_set)}개의 질문이 {OUTPUT_FILE}에 저장되었습니다.")
