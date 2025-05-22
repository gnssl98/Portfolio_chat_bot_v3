import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# === 환경 설정 ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "qa_data_deduplicated.jsonl"
OUTPUT_FILE = "qa_data_translated_fully_en.jsonl"
MODEL_NAME = "gpt-3.5-turbo"

# === GPT 호출 함수 ===
def chat_gpt(messages):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT 호출 실패: {e}")
        return None

# === GPT 응답 JSON 파싱 처리 ===
def safe_parse_json(gpt_response):
    if gpt_response.startswith("```json"):
        gpt_response = gpt_response.replace("```json", "").strip()
    if gpt_response.startswith("```"):
        gpt_response = gpt_response.replace("```", "").strip()
    if gpt_response.endswith("```"):
        gpt_response = gpt_response[:-3].strip()

    try:
        return json.loads(gpt_response)
    except Exception as e:
        print(f"[WARN] JSON 파싱 실패: {e}\n→ GPT 응답:\n{gpt_response}\n")
        return None

# === 하나의 항목 번역
def translate_item(item):
    ko_json = json.dumps(item, ensure_ascii=False)

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a professional translator. "
                "Translate ALL values (not the keys) of the following JSON into fluent and natural English. "
                "Do NOT change the key names. Do NOT generate new answers. "
                "Do NOT leave anything in Korean. Return only valid JSON."
            )
        },
        {
            "role": "user",
            "content": ko_json
        }
    ]

    translated = chat_gpt(prompt)
    if translated is None:
        return None

    parsed = safe_parse_json(translated)
    if parsed:
        return {
            "question": parsed.get("question", "").strip(),
            "context": parsed.get("context", "").strip(),
            "answer": parsed.get("answer", "").strip()
        }
    return None

# === 전체 처리 ===
def main():
    input_data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                input_data.append(json.loads(line.strip()))
            except:
                continue

    output_data = []

    for item in tqdm(input_data, desc="Translating to English"):
        result = translate_item(item)
        if result:
            output_data.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n 번역 완료 → {OUTPUT_FILE} (총 {len(output_data)}개)")

if __name__ == "__main__":
    main()
