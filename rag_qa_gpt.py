import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import BertTokenizer, BertForSequenceClassification, MarianTokenizer, MarianMTModel
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from openai import OpenAI
from typing import TypedDict
import torch

# 환경 설정
load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 경로 설정
VECTOR_PATH = "./vector_store"
KO_EN_MODEL_PATH = r"D:\\dataset\\fine_tuned_model\\translation_model\\ko_en_finetuned"
EN_KO_MODEL_PATH = r"D:\\dataset\\fine_tuned_model\\translation_model\\en_ko_finetuned"

# 임베딩 및 벡터 DB 로딩
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = FAISS.load_local(VECTOR_PATH, embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.5})

# 번역 모델
ko_en_tokenizer = MarianTokenizer.from_pretrained(KO_EN_MODEL_PATH)
ko_en_model = MarianMTModel.from_pretrained(KO_EN_MODEL_PATH).to(device)
en_ko_tokenizer = MBart50TokenizerFast.from_pretrained(EN_KO_MODEL_PATH)
en_ko_model = MBartForConditionalGeneration.from_pretrained(EN_KO_MODEL_PATH).to(device)
en_ko_tokenizer.src_lang = "en_XX"

# 분류 모델
clf_tokenizer = BertTokenizer.from_pretrained("./classifier_model")
clf_model = BertForSequenceClassification.from_pretrained("./classifier_model")
clf_model.eval().to(device)

# 상태 정의
class PortfolioState(TypedDict, total=False):
    question: str
    is_portfolio: bool
    context: str
    answer: str

# 번역 함수
def translate_marian(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=5, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_mbart(text, tokenizer, model, src_lang="en_XX", tgt_lang="ko_KR"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang], max_length=128, num_beams=5, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 질문 리포맷팅 (GPT)
def reformat_question_with_gpt(question_ko: str) -> str:
    prompt = f"""
너는 모호한 한국어 면접 질문을 구체화하는 AI야.
사용자의 질문은 포트폴리오 기반 면접 질문이야.
아래 질문을 더 구체적이고 포트폴리오 기반으로 변환해줘.

질문: {question_ko}
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR] 질문 리포맷 실패:", e)
        return question_ko

# 노드 정의
def classify_with_model(state):
    question = state["question"]
    inputs = clf_tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = clf_model(**inputs).logits
        is_portfolio = torch.argmax(logits, dim=-1).item() == 1
    print("[LOG] classify_with_model 실행됨 → is_portfolio:", is_portfolio)
    return {**state, "is_portfolio": is_portfolio}

def retrieve_context_node(state):
    print("[LOG] retrieve_context_node 실행됨 / state:", state)
    try:
        docs = retriever.invoke(state["question"])

        # "프로젝트" 포함 문서 우선 정렬
        docs = sorted(docs, key=lambda d: "프로젝트" in d.page_content, reverse=True)

        # 최대 3개 문서 선택
        selected = docs[:3]

        context = " ".join([doc.page_content.replace("\n", " ") for doc in selected])
        context = context.strip()[:1000]
        print("[LOG] context 내용:", context)

        return {**state, "context": context}
    except Exception as e:
        print("[ERROR] retrieve_context_node 예외:", e)
        return {**state, "context": ""}

def run_qa_pipeline_node(state):
    print("[LOG] run_qa_pipeline_node 실행됨 / context 존재 여부:", bool(state.get("context")))
    question_ko = state["question"]
    context_ko = state["context"]

    question_en = translate_marian(question_ko, ko_en_tokenizer, ko_en_model)
    context_en = translate_marian(context_ko, ko_en_tokenizer, ko_en_model)

    # GPT 프롬프트 강화
    prompt = f"""
You are a Korean job interview assistant chatbot.

You are given an excerpt from the user's self-introduction portfolio below. The user will ask an interview question in Korean.

Your task:
- Answer the question based ONLY on the portfolio excerpt below.
- Do NOT assume or ask for more information.
- Do NOT say "no information found".
- Provide a concise and formal answer in English.

--- Portfolio Excerpt ---
{context_en}
--------------------------

Interview Question:
Q: {question_en}
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional Korean-English job interview assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer_en = response.choices[0].message.content.strip()
    except Exception as e:
        print("[ERROR] GPT 응답 실패:", e)
        return {**state, "answer": "GPT 응답 생성 실패"}

    print("[LOG] 영어 답변:", answer_en)
    answer_ko = translate_mbart(answer_en, en_ko_tokenizer, en_ko_model)
    print("[LOG] 최종 번역된 답변:", answer_ko)

    return {**state, "answer": answer_ko}

def run_gpt_fallback_node(state):
    print("[LOG] run_gpt_fallback_node 실행됨")
    question = state["question"]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 친절한 한국어 챗봇이야."},
            {"role": "user", "content": question}
        ]
    )
    return {**state, "answer": response.choices[0].message.content.strip()}

# 라우팅 조건
def route_classify(state: PortfolioState) -> str:
    print("[DEBUG] 분기 조건 평가 중 - is_portfolio:", state.get("is_portfolio"))
    return "retrieve" if state.get("is_portfolio", False) else "gpt_fallback"

# LangGraph 빌드
def build_portfolio_graph():
    graph = StateGraph(state_schema=PortfolioState)
    graph.add_node("classify", RunnableLambda(classify_with_model))
    graph.add_node("retrieve", RunnableLambda(retrieve_context_node))
    graph.add_node("qa", RunnableLambda(run_qa_pipeline_node))
    graph.add_node("gpt_fallback", RunnableLambda(run_gpt_fallback_node))
    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", route_classify)
    graph.add_edge("retrieve", "qa")
    graph.add_edge("qa", END)
    graph.add_edge("gpt_fallback", END)
    return graph.compile()

# 실행
dialogue_graph = build_portfolio_graph()

if __name__ == "__main__":
    user_input = input("질문을 입력하세요: ")
    try:
        reformatted = reformat_question_with_gpt(user_input)
        print("[DEBUG] 리포맷된 질문:", reformatted)
        result = dialogue_graph.invoke({"question": reformatted})
        print("[DEBUG] invoke 반환값:", result)
        print("\n답변:", result.get("answer", "응답 없음"))
    except Exception as e:
        print("[ERROR] 전체 실행 중 예외 발생:", e)
