from rag_qa_gpt import dialogue_graph

test_cases = [
    {
        "name": "포트폴리오 관련 질문",
        "question": "포트폴리오에 작성한 프로젝트 중 하나를 설명해주세요.",
        "expected_path": "retrieve → qa"
    },
    {
        "name": "일반 질문",
        "question": "오늘 날씨 어때?",
        "expected_path": "gpt_fallback"
    }
]

for i, case in enumerate(test_cases):
    print(f"\n===== 테스트 {i+1}: {case['name']} =====")
    try:
        result = dialogue_graph.invoke({"question": case["question"]})
        print("[RESULT]", result.get("answer", "응답 없음"))
    except Exception as e:
        print("[ERROR] 예외 발생:", e)
