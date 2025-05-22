import json
from collections import defaultdict

INPUT_FILE = "qa_data.jsonl"

def find_duplicate_questions(path):
    seen = defaultdict(list)  # {question: [index1, index2, ...]}
    
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                question = item.get("question", "").strip()
                seen[question].append(idx)
            except Exception as e:
                print(f"[ERROR] JSONL 파싱 실패: {e}")

    # 중복된 질문만 추출
    duplicates = {q: idxs for q, idxs in seen.items() if len(idxs) > 1}
    
    if duplicates:
        
        for q, idxs in duplicates.items():
            print(f"Q: {q}\n  → 중복 라인 번호: {idxs}")

        print(f"\n🔁 총 {len(duplicates)} 개의 중복된 질문이 발견되었습니다.\n")
    else:
        print("✅ 중복된 질문이 없습니다.")

if __name__ == "__main__":
    find_duplicate_questions(INPUT_FILE)
