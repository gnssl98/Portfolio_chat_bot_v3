import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===== 설정 =====
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_DIR = "./portfolio_data"
OUTPUT_JSONL = "flan_t5_qa_dataset.jsonl"
OUTPUT_JSON = "flan_t5_qa_dataset.json"

QUESTION_LIST = [
  "자기소개를 해주세요.",
  "본인을 소개해 주세요.",
  "당신은 어떤 사람인가요?",
  "자신에 대해 설명해 주세요.",
  "당신의 강점을 말해 주세요.",
  "자신의 장점은 무엇인가요?",
  "어떤 역량이 가장 자신 있나요?",
  "단점이 있다면 무엇인가요?",
  "자신의 약점을 어떻게 보완하나요?",
  "가장 기억에 남는 프로젝트는 무엇인가요?",
  "참여한 프로젝트 중 가장 인상 깊었던 것은?",
  "프로젝트 경험에 대해 말씀해 주세요.",
  "프로젝트 경험이 있나요?",
  "프로젝트를 수행한 경험을 설명해 주세요.",
  "팀 프로젝트 경험이 있나요?",
  "혼자서 진행한 프로젝트가 있다면 소개해 주세요.",
  "AI 관련 프로젝트를 수행한 경험이 있나요?",
  "가장 도전적이었던 경험은 무엇인가요?",
  "문제를 해결한 경험을 설명해 주세요.",
  "문제 해결 능력을 발휘한 사례가 있나요?",
  "리더십을 발휘한 경험이 있다면?",
  "팀워크를 발휘한 경험이 있다면 설명해 주세요.",
  "갈등을 해결한 경험이 있나요?",
  "성공적으로 마무리한 과제를 설명해 주세요.",
  "실패했던 경험이 있다면?",
  "실패를 통해 배운 점이 있다면 무엇인가요?",
  "기술적인 어려움을 극복한 경험을 알려 주세요.",
  "AI 분야에 관심을 가지게 된 계기는?",
  "AI 연구에 흥미를 느끼게 된 이유는?",
  "컴퓨터공학을 전공하게 된 이유는?",
  "해당 전공을 선택한 이유는 무엇인가요?",
  "대학 생활 중 기억에 남는 활동은?",
  "가장 인상 깊었던 강의나 교수님은 누구인가요?",
  "학업 외 활동 중 자랑하고 싶은 것이 있다면?",
  "인턴 경험이 있나요?",
  "산학 협력 프로젝트에 참여한 적 있나요?",
  "졸업 프로젝트에 대해 설명해 주세요.",
  "졸업 논문이나 결과물에 대해 말해 주세요.",
  "자격증이 있다면 소개해 주세요.",
  "어떤 자격증을 보유하고 있나요?",
  "학습을 위해 어떤 노력을 하나요?",
  "자기계발을 위해 어떤 활동을 하나요?",
  "최근에 공부한 기술이나 개념은 무엇인가요?",
  "관심 있는 기술 분야는 무엇인가요?",
  "앞으로 배우고 싶은 기술은 무엇인가요?",
  "협업을 위한 도구나 환경을 사용해 본 경험이 있나요?",
  "버전 관리 시스템을 사용한 경험이 있나요?",
  "Git이나 GitHub를 사용할 줄 아시나요?",
  "코드 리뷰 경험이 있다면 설명해 주세요.",
  "개발 관련 블로그나 포트폴리오가 있나요?",
  "기술 블로그를 운영해 본 경험이 있나요?",
  "오픈소스에 기여한 경험이 있나요?",
  "개발 커뮤니티 활동 경험이 있나요?",
  "당신의 목표는 무엇인가요?",
  "장기적인 커리어 계획은 어떻게 되나요?",
  "10년 뒤 본인의 모습은 어떨 것 같나요?",
  "입사 후 어떤 목표를 이루고 싶나요?",
  "우리 회사에 지원한 이유는 무엇인가요?",
  "왜 이 직무에 관심을 가지게 되었나요?",
  "지원 동기를 말씀해 주세요.",
  "회사에 기여할 수 있는 점은 무엇인가요?",
  "해당 직무에 강점이 있다고 생각하는 이유는?",
  "비슷한 업무를 수행한 경험이 있나요?",
  "자기 주도적으로 일한 경험이 있나요?",
  "책임감을 가지고 일한 경험을 말해 주세요.",
  "시간을 효율적으로 관리한 경험이 있나요?",
  "마감 기한을 맞추기 위해 노력한 경험이 있나요?",
  "스트레스를 어떻게 관리하나요?",
  "압박 속에서도 능력을 발휘한 경험이 있다면?",
  "당신이 생각하는 좋은 팀원이란 무엇인가요?",
  "협업 시 가장 중요하게 생각하는 점은?",
  "팀원 간 의견 충돌이 있을 때 어떻게 하나요?",
  "당신이 다른 지원자보다 뛰어난 점은?",
  "본인의 차별화된 경쟁력은 무엇인가요?",
  "자신이 성실하다고 생각하나요?",
  "자신의 커뮤니케이션 스타일은 어떤가요?",
  "문서를 정리하거나 공유하는 데 강점이 있나요?",
  "기술을 쉽게 설명할 수 있나요?",
  "자기 소개서에서 강조하고 싶은 부분은?",
  "자기소개서를 직접 작성했나요?",
  "자기소개서에서 빠진 내용이 있다면 무엇인가요?",
  "추가로 소개하고 싶은 경험이 있나요?",
  "가장 최근에 수행한 프로젝트는 무엇인가요?",
  "마지막으로 하고 싶은 말이 있다면?",
  "자기소개 외에 강조하고 싶은 것이 있다면?",
  "학부에서 배운 과목 중 가장 유용했던 것은?",
  "자신이 참여한 공모전이나 대회가 있다면 소개해 주세요.",
  "연구 경험이 있다면 구체적으로 설명해 주세요.",
  "논문이나 리포트를 작성한 경험이 있나요?",
  "실제 사용자와 함께 일해 본 경험이 있나요?",
  "사용자 피드백을 반영해 개선한 경험이 있다면?",
  "배운 내용을 실전에 적용한 경험을 말해 주세요.",
  "학습한 내용을 팀원과 공유해 본 적이 있나요?",
  "문제 상황에서 침착하게 대응한 사례가 있다면?",
  "기술 트렌드에 대해 어떻게 파악하고 있나요?",
  "최근 주목하는 기술 동향은 무엇인가요?",
  "당신의 기술 스택을 설명해 주세요."
]


# ===== 문서 로딩 =====
def load_documents():
    documents = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(path)
                elif ext == ".txt":
                    loader = TextLoader(path, encoding="utf-8")
                else:
                    continue
                docs = loader.load()
                documents.extend(docs)
                print(f"[LOADED] {file} → {len(docs)} docs")
            except Exception as e:
                print(f"[SKIPPED] {file} → Error: {e}")
    return documents

# ===== 문서 분할 =====
def split_documents(docs, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(docs)

# ===== GPT 호출 함수 =====
def generate_answer(context, question):
    context = context.strip()
    if len(context) < 100:
        return None
    if "전자증명서" in context or "출력물은 사본" in context:
        return None

    system = "You are a helpful assistant answering questions based on a person's portfolio."
    user = f"""Use the following portfolio content to answer the question below.

Context:
{context}

Question: {question}

Answer in a concise, professional, and accurate tone. Base your answer only on the context."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR GPT]", e)
        return None

# ===== 실행 =====
if __name__ == "__main__":
    raw_docs = load_documents()
    print("문서 수:", len(raw_docs))

    if not raw_docs:
        print("⚠️ PDF/TXT 문서를 찾지 못했습니다.")
        exit()

    chunks = split_documents(raw_docs)
    print("분할된 청크 수:", len(chunks))

    qa_data = []

    for i, doc in enumerate(chunks):
        context = doc.page_content.strip()

        for q in random.sample(QUESTION_LIST, min(5, len(QUESTION_LIST))):  # 질문당 5개 생성
            print(f"[{i+1}] 질문: {q}")
            answer = generate_answer(context, q)
            if answer:
                qa_data.append({
                    "input": f"Answer the following question based on the context.\n\nContext: {context}\n\nQuestion: {q}",
                    "output": answer
                })


    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for item in qa_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    print(f"총 생성된 QA 쌍 수: {len(qa_data)}")
