import os
import json
import time
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# ===== 모델 경로 =====
KO_EN_MODEL_PATH = r"D:\dataset\fine_tuned_model\translation_model\ko_en_finetuned"

# ===== 데이터 경로 =====
INPUT_FILE = "qa_data_deduplicated.jsonl"
OUTPUT_FILE = "qa_data_translated.jsonl"

# ===== 로컬 번역기 불러오기 =====
def load_translation_model(path):
    tokenizer = MarianTokenizer.from_pretrained(path)
    model = MarianMTModel.from_pretrained(path).to("cuda")
    return tokenizer, model

# ===== 배치 번역 함수 (질문+문맥+답변 포함 전체를 한 번에 처리) =====
def batch_translate(texts, tokenizer, model, max_length=384):
    if not texts:
        return []
    
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to("cuda")
    
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=1,  # greedy decoding for speed
        early_stopping=True
    )
    elapsed = time.time() - start_time

    # 모니터링 출력
    print(f"번역 시간: {elapsed:.2f}s | avg {elapsed/len(texts):.2f}s/sample | max_len: {max_length} | batch: {len(texts)}")

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# ===== 실행 =====
def main():
    tokenizer, model = load_translation_model(KO_EN_MODEL_PATH)

    batch_size = 8
    data = []

    # 데이터 로딩
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except Exception as e:
                print(f"[ERROR] JSON 파싱 실패: {e}")

    translated_data = []

    for i in tqdm(range(0, len(data), batch_size), desc="Translating", ncols=80):
        batch = data[i:i+batch_size]

        texts_to_translate = []
        for ex in batch:
            texts_to_translate.extend([ex["question"], ex["context"], ex["answer"]])

        try:
            translated = batch_translate(texts_to_translate, tokenizer, model)
        except Exception as e:
            print(f"[ERROR] 번역 실패 at batch {i}: {e}")
            continue

        for j in range(0, len(translated), 3):
            translated_data.append({
                "question": translated[j],
                "context": translated[j+1],
                "answer": translated[j+2]
            })

    # 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in translated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n 총 {len(translated_data)}개 항목 번역 완료 → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
