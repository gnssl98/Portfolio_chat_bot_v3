import json
from collections import defaultdict

INPUT_FILE = "qa_data.jsonl"
OUTPUT_FILE = "qa_data_deduplicated.jsonl"

def deduplicate_questions(input_path, output_path):
    question_groups = defaultdict(list)

    # 1. 질문별로 그룹화
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                question = item.get("question", "").strip()
                question_groups[question].append(item)
            except Exception as e:
                print(f"[ERROR] JSONL 파싱 실패: {e}")

    # 2. 각 그룹에서 answer가 가장 긴 항목 선택
    unique_data = []
    for q, group in question_groups.items():
        best = max(group, key=lambda x: len(x.get("answer", "")))
        unique_data.append(best)

    # 3. 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for item in unique_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 총 {len(unique_data)} 개의 질문이 남았습니다. 중복 제거 완료 → {output_path}")

if __name__ == "__main__":
    deduplicate_questions(INPUT_FILE, OUTPUT_FILE)
