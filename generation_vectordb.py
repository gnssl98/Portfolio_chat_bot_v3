import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ===== 설정 =====
DATA_DIR = "./portfolio_data"
VECTOR_DIR = "./vector_store"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# ===== 1. 문서 로딩 =====
def load_documents(data_dir):
    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif file.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                else:
                    continue
                docs = loader.load()
                documents.extend(docs)
                print(f"[LOADED] {file} → {len(docs)} docs")
            except Exception as e:
                print(f"[ERROR] {file} 로드 실패: {e}")
    print(f"[TOTAL] 총 {len(documents)}개 문서 로딩 완료")
    return documents

# ===== 2. 문서 분할 =====
def split_documents(documents, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"[INFO] 총 {len(chunks)}개의 청크로 분할 완료")
    return chunks

# ===== 3. 벡터 DB 생성 및 저장 =====
def build_vector_store(docs, model_name, save_dir):
    if not docs:
        print("[WARNING] 처리할 문서가 없습니다")
        return

    os.makedirs(save_dir, exist_ok=True)
    embedding = HuggingFaceEmbeddings(model_name=model_name)

    try:
        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local(save_dir)
        print(f"[SUCCESS] 벡터 저장소 저장 완료 → {save_dir}")
    except Exception as e:
        print(f"[ERROR] 벡터 저장소 생성 실패: {e}")

# ===== 실행 =====
if __name__ == "__main__":
    print("문서 로딩 시작...")
    raw_docs = load_documents(DATA_DIR)

    if not raw_docs:
        print("문서를 불러오지 못했습니다.")
        exit()

    print("\n문서 분할 중...")
    chunks = split_documents(raw_docs)

    if not chunks:
        print("문서 분할 실패.")
        exit()

    print("\n 벡터 저장소 생성 중...")
    build_vector_store(chunks, EMBEDDING_MODEL_NAME, VECTOR_DIR)
