import json

ORIGINAL_FILE = "rag_generated_qa.jsonl"
REFINED_FILE = "refined_info.jsonl"
OUTPUT_FILE = "qa_data.jsonl"
TARGET_ANSWER = "정보가 부족합니다"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                data.append(obj)
            except Exception as e:
                print(f"[ERROR] {path} 라인 파싱 실패: {e}")
    return data

def merge_filtered_and_refined(original_path, refined_path, output_path):
    # 1. 원본 데이터 중 '정보가 부족합니다' 아닌 것만 선택
    original_data = load_jsonl(original_path)
    filtered = [ex for ex in original_data if ex.get("answer", "").strip() != TARGET_ANSWER]
    print(f"✅ 원본 중 정상 QA: {len(filtered)}개")

    # 2. 보완된 데이터 로드
    refined = load_jsonl(refined_path)
    print(f"✅ 보완된 QA: {len(refined)}개")

    # 3. 병합
    merged = filtered + refined

    # 4. 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 총 {len(merged)} 개 QA가 저장됨 → {output_path}")

if __name__ == "__main__":
    merge_filtered_and_refined(ORIGINAL_FILE, REFINED_FILE, OUTPUT_FILE)
