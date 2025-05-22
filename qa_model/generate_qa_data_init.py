import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ================== 환경 설정 ==================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "../portfolio_data"
QUESTION_FILE = "questions.jsonl"
VECTOR_DIR = "./vector_store"
OUTPUT_JSONL = "rag_generated_qa.jsonl"
OUTPUT_JSON = "rag_generated_qa.json"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# ================== 문서 로딩 ==================
def load_documents(data_dir):
    documents = []
    for root, _, files in os.walk(data_dir):
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
                print(f"[LOADED] {os.path.relpath(path, data_dir)} → {len(docs)} docs")
            except Exception as e:
                print(f"[ERROR] {file}: {e}")
    print(f"✅ 총 {len(documents)} 개 문서를 로드했습니다.")
    return documents

# ================== 문단 분할 ==================
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# ================== 질문 로딩 ==================
def load_questions(filepath):
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                q = json.loads(line.strip())
                if "text" in q:
                    questions.append(q["text"])
            except:
                pass
    print(f"✅ 총 {len(questions)} 개 질문 로드 완료")
    return questions

# ================== VectorDB 생성 ==================
def create_vector_db(chunks, save_path=None):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    if save_path:
        vectordb.save_local(save_path)
        print(f"💾 Vector DB 저장됨: {save_path}")
    return vectordb

# ================== GPT로 답변 생성 ==================
def generate_answer_from_rag(question, retrieved_docs):
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    system_prompt = (
        "You are a helpful assistant answering questions about the user's portfolio using the context provided.\n"
        "Answer concisely using the information. If the context is insufficient, say '정보가 부족합니다'."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=messages,
            temperature=0.7,
        )
        return context, response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] GPT 응답 실패: {e}")
        return context, "정보가 부족합니다"

# ================== 전체 파이프라인 ==================
def main():
    # 문서 처리
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    vectordb = create_vector_db(chunks, save_path=VECTOR_DIR)

    # 질문 처리
    questions = load_questions(QUESTION_FILE)

    # 답변 생성
    qa_results = []
    for question in questions:
        docs = vectordb.similarity_search(question, k=3)
        context, answer = generate_answer_from_rag(question, docs)
        qa_results.append({
            "question": question,
            "context": context,
            "answer": answer
        })
        print(f"✅ Q: {question}\n→ A: {answer}\n")

    # 저장
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for qa in qa_results:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(qa_results, f, ensure_ascii=False, indent=2)

    print(f"\n총 {len(qa_results)} 개 QA 결과 저장 완료")

if __name__ == "__main__":
    main()
