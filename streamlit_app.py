import streamlit as st
from rag import dialogue_graph  # rag.py에서 정의한 LangGraph 파이프라인 사용

st.set_page_config(page_title="자기소개 RAG 챗봇", layout="centered")

st.title("📄 자기소개 포트폴리오 챗봇")
st.write("자기소개서, 이력서, 포트폴리오 기반으로 질문에 답변해드립니다.")

# 사용자 질문 입력
question = st.text_input("질문을 입력하세요:", placeholder="예: 전공은 무엇인가요?")

if st.button("질문하기") and question.strip():
    with st.spinner("답변 생성 중..."):
        try:
            result = dialogue_graph.invoke({"question": question})
            st.success("답변:")
            st.markdown(f"**{result['answer']}**")
        except Exception as e:
            st.error("오류가 발생했습니다.")
            st.exception(e)
