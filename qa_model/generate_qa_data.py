import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Deprecated class import 수정
import pickle

# 설정
DATA_DIR = "./portfolio_data"
VECTOR_DIR = "./vector_store"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# 1. 문서 로딩
def load_documents(data_dir):
    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif file.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                continue
            documents.extend(loader.load())
    return documents

# 2. 문서 분할
def split_documents(documents, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# 3. 임베딩 및 벡터 DB 생성
def build_vector_store(docs, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # HuggingFaceEmbeddings 모델 로딩 및 벡터화
    embedding = HuggingFaceEmbeddings(model_name=model_name)  # Embedding 모델 지정
    vectorstore = FAISS.from_documents(docs, embedding)  # FAISS 벡터 DB 생성

    # 벡터 DB 저장
    vectorstore.save_local(save_dir)
    print(f"벡터 저장소 저장 완료: {save_dir}")

# 실행
if __name__ == "__main__":
    print("문서 로딩 중...")
    raw_docs = load_documents(DATA_DIR)

    print(f"{len(raw_docs)}개 문서를 chunk로 분할 중...")
    chunks = split_documents(raw_docs)

    print("벡터 저장소 생성 중...")
    build_vector_store(chunks, EMBEDDING_MODEL_NAME, VECTOR_DIR)
