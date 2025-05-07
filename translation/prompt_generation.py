import os
import json
import openai
from openai import OpenAI
from tqdm import tqdm
from langdetect import detect
from dotenv import load_dotenv

load_dotenv()  # .env 파일 로드

DOCUMENT_FILE = 'document.txt'
OUTPUT_JSONL = 'translation_gpt_pairs.jsonl'
OUTPUT_JSON = 'translation_gpt_pairs.json'

# 문장 단위 로딩
def load_sentences(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('===== 파일:')]

# GPT 번역 요청
def generate_translation(src_text, direction="en->ko"):
    if direction == "en->ko":
        system = "You're a professional English-to-Korean translator."
        user = f"Translate the following sentence into Korean:\n\n{src_text}"
    else:
        system = "You're a professional Korean-to-English translator."
        user = f"다음 문장을 자연스럽게 영어로 번역해줘:\n\n{src_text}"

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        print(f"❌ GPT 에러: {e}")
        return None

# 문장 쌍 생성
def create_translation_dataset(sentences):
    examples = []

    for sent in tqdm(sentences):
        try:
            lang = detect(sent)
        except:
            continue  # 언어 감지 실패 시 skip

        if lang == "en":
            translated = generate_translation(sent, direction="en->ko")
            if translated:
                examples.append({"translation": {"en": sent, "ko": translated}})
        elif lang == "ko":
            translated = generate_translation(sent, direction="ko->en")
            if translated:
                examples.append({"translation": {"ko": sent, "en": translated}})
        else:
            continue

    return examples

if __name__ == "__main__":
    print("document.txt 문장 로딩 중...")
    sentences = load_sentences(DOCUMENT_FILE)

    print("GPT로 번역 중...")
    examples = create_translation_dataset(sentences)

    print("JSONL 저장 중...")
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in examples:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"\n문장 쌍 기반 번역 데이터 생성 완료!\n- {OUTPUT_JSONL}\n- {OUTPUT_JSON}")
