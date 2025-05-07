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

# ====== 벡터 저장소 로딩 ======
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = FAISS.load_local(VECTOR_PATH, embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# ====== QA 모델 로딩 ======
qa_tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(QA_MODEL_PATH)
qa_pipeline = pipeline("text2text-generation", model=qa_model, tokenizer=qa_tokenizer, max_new_tokens=128)
qa_chain = HuggingFacePipeline(pipeline=qa_pipeline)

# 한국어 → 영어 번역 모델 로딩
ko_en_tokenizer = MBart50TokenizerFast.from_pretrained(KO_EN_MODEL_PATH)
ko_en_model = MBartForConditionalGeneration.from_pretrained(KO_EN_MODEL_PATH)

# 영어 → 한국어 번역 모델 로딩
en_ko_tokenizer = MBart50TokenizerFast.from_pretrained(EN_KO_MODEL_PATH)
en_ko_model = MBartForConditionalGeneration.from_pretrained(EN_KO_MODEL_PATH)


def translate(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== GPT API 클라이언트 ======
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== 분류 모델 로딩 ======
clf_tokenizer = BertTokenizer.from_pretrained("./classifier_model")
clf_model = BertForSequenceClassification.from_pretrained("./classifier_model")
clf_model.eval().cuda()

class PortfolioState(TypedDict, total=False):
    question: str
    is_portfolio: bool
    context: str
    answer: str
    answer_en: str
    validated: bool
    search_attempts: int

# ====== 노드 정의 ======
def classify_with_model(state):
    question = state["question"]
    inputs = clf_tokenizer(question, return_tensors="pt", truncation=True, padding=True).to("cuda")
    with torch.no_grad():
        logits = clf_model(**inputs).logits
        is_portfolio = torch.argmax(logits, dim=-1).item() == 1
    print("[LOG] classify_with_model 실행됨 → is_portfolio:", is_portfolio)
    return {**state, "is_portfolio": is_portfolio, "search_attempts": 0}

def retrieve_context_node(state):
    print("[LOG] retrieve_context_node 실행됨")
    docs = retriever.get_relevant_documents(state["question"])
    context = "\n".join([doc.page_content for doc in docs])
    return {**state, "context": context}

def run_qa_node(state):
    print("[LOG] run_qa_node 실행됨")
    question_ko = state["question"]
    question_en = translate(question_ko, ko_en_tokenizer, ko_en_model)
    print("[LOG] 번역된 질문:", question_en)
    
    prompt = PromptTemplate.from_template("""
    Below is information extracted from a self-introduction portfolio document:
    -----------
    {context}
    -----------
    Based on the above, answer the following question:
    Question: {question}
    """)
    final_prompt = prompt.format(question=question_en, context=state["context"])
    answer_en = qa_chain.invoke(final_prompt)
    return {**state, "answer_en": answer_en}

def validate_answer_node(state):
    answer_en = state.get("answer_en", "")
    attempts = state.get("search_attempts", 0)
    is_valid = answer_en and len(answer_en.strip()) >= 5 and answer_en.strip().lower() not in ["i don't know", "unknown"]
    print(f"[LOG] validate_answer_node 실행됨 → validated: {is_valid}, attempts: {attempts}")

    if not answer_en:
        return {**state, "answer": "죄송합니다. 적절한 답변을 생성하지 못했습니다.", "validated": True}

    answer_ko = translate(answer_en, en_ko_tokenizer, en_ko_model)
    print("[LOG] 번역된 답변:", answer_ko)
    
    state["answer"] = answer_ko

    if not is_valid:
        if attempts >= 2:
            return {**state, "validated": True}
        else:
            return {**state, "validated": False, "search_attempts": attempts + 1}
    return {**state, "validated": True}

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

# ====== LangGraph 구성 ======
def build_portfolio_graph():
    graph = StateGraph(state_schema=PortfolioState)
    graph.add_node("classify", RunnableLambda(classify_with_model))
    graph.add_node("retrieve", RunnableLambda(retrieve_context_node))
    graph.add_node("qa", RunnableLambda(run_qa_node))
    graph.add_node("validate", RunnableLambda(validate_answer_node))
    graph.add_node("gpt_fallback", RunnableLambda(run_gpt_fallback_node))

    graph.set_entry_point("classify")
    graph.add_conditional_edges("classify", {
        "retrieve": lambda x: x["is_portfolio"],
        "gpt_fallback": lambda x: not x["is_portfolio"]
    })
    graph.add_edge("retrieve", "qa")
    graph.add_edge("qa", "validate")
    graph.add_conditional_edges("validate", {
        END: lambda x: x["validated"],
        "retrieve": lambda x: not x["validated"]
    })
    graph.add_edge("gpt_fallback", "validate")
    return graph.compile()

# ====== 실행 ======
dialogue_graph = build_portfolio_graph()

if __name__ == "__main__":
    user_input = input("질문을 입력하세요: ")
    result = dialogue_graph.invoke({"question": user_input})
    if "answer" in result:
        print("\n 답변:", result["answer"])
    else:
        print("\n 답변을 생성하지 못했습니다.")
