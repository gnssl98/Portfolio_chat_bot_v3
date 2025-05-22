import json

input_path = "classification_data.jsonl"
output_path = "label_1_only.jsonl"

filtered = []

with open(input_path, "r", encoding="utf-8") as infile:
    for idx, line in enumerate(infile, start=1):
        line = line.strip()
        if not line:
            continue  # 빈 줄 건너뜀
        try:
            item = json.loads(line)
            if item.get("label") == 1:
                filtered.append(item)
        except json.JSONDecodeError as e:
            print(f"[SKIP] JSON 오류 (line {idx}): {e}")
            continue

with open(output_path, "w", encoding="utf-8") as outfile:
    for item in filtered:
        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 'label': 1인 항목 {len(filtered)}개 저장 완료 → {output_path}")
