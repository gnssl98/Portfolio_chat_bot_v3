import json

INPUT_FILE = "rag_generated_qa.jsonl"
OUTPUT_FILE = "info.jsonl"
TARGET_ANSWER = "정보가 부족합니다."

def extract_info_only(input_path, output_path, target_answer):
    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                if data.get("answer", "").strip() == target_answer:
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                    count += 1
            except Exception as e:
                print(f"[ERROR] Invalid JSONL line: {e}")
    print(f"총 {count} 개의 '정보가 부족합니다' 답변이 저장되었습니다 → {output_path}")

if __name__ == "__main__":
    extract_info_only(INPUT_FILE, OUTPUT_FILE, TARGET_ANSWER)
