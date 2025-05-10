import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import BertTokenizer, BertForSequenceClassification, MarianTokenizer, MarianMTModel
from langchain.prompts import PromptTemplate
from openai import OpenAI
from typing import TypedDict
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

load_dotenv()

# ====== 경로 설정 ======
VECTOR_PATH = "./vector_store"
KO_EN_MODEL_PATH = r"D:\\dataset\\fine_tuned_model\\translation_model\\ko_en_finetuned"
EN_KO_MODEL_PATH = r"D:\\dataset\\fine_tuned_model\\translation_model\\en_ko_finetuned"
QA_MODEL_PATH = r"D:\\dataset\\fine_tuned_model\\flan-t5-large"

# ====== 디바이스 설정 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 벡터 저장소 로딩 ======
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = FAISS.load_local(VECTOR_PATH, embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# ====== QA 모델 로딩 ======
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL_PATH).to(device)
qa_pipeline = pipeline("text2text-generation", model=qa_model, tokenizer=qa_tokenizer, max_new_tokens=128, device=0 if torch.cuda.is_available() else -1)
qa_chain = HuggingFacePipeline(pipeline=qa_pipeline)

# ====== 번역 모델 로딩 ======
ko_en_tokenizer = MarianTokenizer.from_pretrained(KO_EN_MODEL_PATH)
ko_en_model = MarianMTModel.from_pretrained(KO_EN_MODEL_PATH).to(device)

en_ko_tokenizer = MBart50TokenizerFast.from_pretrained(EN_KO_MODEL_PATH)
en_ko_model = MBartForConditionalGeneration.from_pretrained(EN_KO_MODEL_PATH).to(device)
en_ko_tokenizer.src_lang = "en_XX"

# ====== 번역 함수 정의 ======
def translate_marian(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_mbart(text, tokenizer, model, src_lang="en_XX", tgt_lang="ko_KR"):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=128,
            num_beams=5,
            no_repeat_ngram_size=2
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== GPT API 클라이언트 ======
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== 분류 모델 로딩 ======
clf_tokenizer = BertTokenizer.from_pretrained("./classifier_model")
clf_model = BertForSequenceClassification.from_pretrained("./classifier_model")
clf_model.eval().cuda()

# ====== 상태 정의 ======
class PortfolioState(TypedDict, total=False):
    question: str
    is_portfolio: bool
    context: str
    answer: str

# ====== 노드 정의 ======
def classify_with_model(state):
    question = state["question"]
    inputs = clf_tokenizer(question, return_tensors="pt", truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        logits = clf_model(**inputs).logits
        is_portfolio = torch.argmax(logits, dim=-1).item() == 1
    print("[LOG] classify_with_model 실행됨 → is_portfolio:", is_portfolio)
    print("[LOG] 반환 state:", {**state, "is_portfolio": is_portfolio})
    return {**state, "is_portfolio": is_portfolio}

def retrieve_context_node(state):
    print("[LOG] retrieve_context_node 실행됨 / state:", state)
    try:
        docs = retriever.get_relevant_documents(state["question"])
        print("[LOG] 관련 문서 개수:", len(docs))
        context = "\n".join([doc.page_content for doc in docs])
        context = context.replace("\n", " ").strip()[:1000]  # 🧹 전처리: 줄바꿈 제거 + 길이 제한
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

    prompt = PromptTemplate.from_template("""
    You are a professional assistant that answers interview questions based on the applicant's portfolio.

    Below is the applicant's self-introduction document:
    -----------
    {context}
    -----------
    Now answer the following interview question in a concise and professional manner:
    Q: {question}
    """)
    final_prompt = prompt.format(question=question_en, context=context_en)
    answer_en = qa_chain.invoke(final_prompt)

    if isinstance(answer_en, dict):
        answer_en = answer_en.get("generated_text", "")

    print("[LOG] 영어 답변:", answer_en)

    answer_ko = translate_mbart(answer_en, en_ko_tokenizer, en_ko_model, src_lang="en_XX", tgt_lang="ko_KR")
    print("[LOG] 최종 번역된 답변:", answer_ko)

    return {**state, "answer": answer_ko}

def run_gpt_fallback_node(state):
    print("[LOG] run_gpt_fallback_node 실행됨")
    question = state["question"]
    messages = [
        {"role": "system", "content": "너는 친절한 챗봇이야."},
        {"role": "user", "content": question}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return {**state, "answer": response.choices[0].message.content.strip()}

# ====== 분기 조건 함수 ======
def route_classify(state: PortfolioState) -> str:
    print("[DEBUG] 분기 조건 평가 중 - is_portfolio:", state.get("is_portfolio"))
    return "retrieve" if state.get("is_portfolio", False) else "gpt_fallback"

# ====== LangGraph 구성 ======
def build_portfolio_graph():
    graph = StateGraph(state_schema=PortfolioState)

    graph.add_node("classify", RunnableLambda(classify_with_model))
    print("[DEBUG] classify 노드 등록")
    graph.add_node("retrieve", RunnableLambda(retrieve_context_node))
    print("[DEBUG] retrieve 노드 등록")
    graph.add_node("qa", RunnableLambda(run_qa_pipeline_node))
    print("[DEBUG] qa 노드 등록")
    graph.add_node("gpt_fallback", RunnableLambda(run_gpt_fallback_node))
    print("[DEBUG] gpt_fallback 노드 등록")

    graph.set_entry_point("classify")
    print("[DEBUG] entry point 설정")

    graph.add_conditional_edges("classify", route_classify)
    print("[DEBUG] 분기 조건 함수 적용 완료")

    graph.add_edge("retrieve", "qa")
    graph.add_edge("qa", END)
    graph.add_edge("gpt_fallback", END)
    print("[DEBUG] 그래프 엣지 설정 완료")

    return graph.compile()

# ====== 실행 ======
dialogue_graph = build_portfolio_graph()

if __name__ == "__main__":
    user_input = input("질문을 입력하세요: ")
    try:
        result = dialogue_graph.invoke({"question": user_input})
        print("[DEBUG] invoke 반환값:", result)
        if "answer" in result and result["answer"]:
            print("\n 답변:", result["answer"])
        else:
            print("\n 답변을 생성하지 못했습니다.")
    except Exception as e:
        print("[ERROR] 전체 실행 중 예외 발생:", e)
