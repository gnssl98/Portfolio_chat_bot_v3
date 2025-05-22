import json
from langdetect import detect, LangDetectException

TRANSLATED_FILE = "qa_data_translated.jsonl"

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def check_translation_quality(filepath):
    total = 0
    empty_question = 0
    empty_context = 0
    empty_answer = 0

    short_question = 0
    short_answer = 0

    non_en_question = 0
    non_en_context = 0
    non_en_answer = 0

    print("\n🔎 영어가 아닌 항목들:")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            try:
                item = json.loads(line.strip())
                q = item.get("question", "").strip()
                c = item.get("context", "").strip()
                a = item.get("answer", "").strip()

                # 비어 있는 항목
                if not q:
                    empty_question += 1
                if not c:
                    empty_context += 1
                if not a:
                    empty_answer += 1

                # 짧은 항목
                if len(q) < 5:
                    short_question += 1
                if len(a) < 10:
                    short_answer += 1

                # 언어 검사
                if q:
                    q_lang = detect_language(q)
                    if q_lang != "en":
                        non_en_question += 1
                        print(f"[{line_num}] Q ({q_lang}): {q}")

                if c:
                    c_lang = detect_language(c)
                    if c_lang != "en":
                        non_en_context += 1
                        print(f"[{line_num}] C ({c_lang}): {c[:100]}{'...' if len(c) > 100 else ''}")

                if a:
                    a_lang = detect_language(a)
                    if a_lang != "en":
                        non_en_answer += 1
                        print(f"[{line_num}] A ({a_lang}): {a}")

            except Exception as e:
                print(f"[ERROR] JSON 파싱 실패 (line {line_num}): {e}")

    print("\n🧪 번역 품질 검사 결과:")
    print(f"총 항목: {total}")
    print(f" - 빈 질문: {empty_question}")
    print(f" - 빈 문맥: {empty_context}")
    print(f" - 빈 답변: {empty_answer}")
    print(f" - 짧은 질문 (<5자): {short_question}")
    print(f" - 짧은 답변 (<10자): {short_answer}")
    print(f" - 영어가 아닌 질문: {non_en_question}")
    print(f" - 영어가 아닌 문맥: {non_en_context}")
    print(f" - 영어가 아닌 답변: {non_en_answer}")

if __name__ == "__main__":
    check_translation_quality(TRANSLATED_FILE)
