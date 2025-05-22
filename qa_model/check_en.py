import json
import re

INPUT_FILE = "qa_data_translated_fully_en.jsonl"
OUTPUT_FILE = "qa_data_filtered_soft_korean.jsonl"

# 10자 이상 연속된 한글 → 긴 문장성 한글로 간주
LONG_KOREAN_PATTERN = re.compile(r"[가-힣]{5,}")

def contains_long_korean(text):
    return bool(LONG_KOREAN_PATTERN.search(text))

def is_clean(entry):
    return not (
        contains_long_korean(entry.get("question", "")) or
        contains_long_korean(entry.get("context", "")) or
        contains_long_korean(entry.get("answer", ""))
    )

def main():
    cleaned = []
    total = 0
    removed = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                entry = json.loads(line.strip())
                if is_clean(entry):
                    cleaned.append(entry)
                else:
                    removed += 1
                    print(f"[REMOVED] 긴 한국어 포함 → Q: {entry['question'][:50]}")
            except:
                continue

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in cleaned:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n 필터링 완료: 총 {total}개 중 {removed}개 제거 → 남은 {len(cleaned)}개 저장됨 → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
