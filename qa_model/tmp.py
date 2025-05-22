import json

LABEL_FILE = "label_1_only.jsonl"  # 기존 데이터셋 파일
OUTPUT_JSONL = "questions.jsonl"  # 추출된 질문을 저장할 파일

# 1. label_1_only.jsonl에서 질문만 추출하여 저장
def extract_questions(label_file, output_jsonl):
    questions = []
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())  # JSON 파싱
                question = data.get("text")  # "text" 키에서 질문 추출
                if question:  # 질문이 있을 경우
                    questions.append({"text": question})
        
        # 추출된 질문을 새로운 jsonl 파일로 저장
        with open(output_jsonl, "w", encoding="utf-8") as f_out:
            for question in questions:
                f_out.write(json.dumps(question, ensure_ascii=False) + "\n")
        
        print(f"질문이 {len(questions)}개 추출되어 {output_jsonl}에 저장되었습니다.")
    
    except Exception as e:
        print(f"[ERROR] Failed to process file: {e}")

# 실행
if __name__ == "__main__":
    extract_questions(LABEL_FILE, OUTPUT_JSONL)
