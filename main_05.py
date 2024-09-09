# main_05 : pdf파일 읽기 + splitting + embedding + db 저장 + 챗봇 기능 추가+ web 서비스
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import os
import streamlit as st
import tempfile

# Chroma DB가 저장된 디렉토리 경로
persist_directory = './db/chromadb'

# Streamlit 웹페이지 제목 설정
st.title("ChatPDF")
st.write("---")

# 파일 업로드 기능 구현
uploaded_file = st.file_uploader("Choose a file")  # 파일 업로드 위젯 생성
st.write("---")

# PDF 파일을 처리하는 함수 (PDF를 임시 폴더에 저장 후 페이지별로 로드)
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()  # 임시 디렉토리 생성
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)  # 임시 파일 경로 설정
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())  # 업로드된 파일을 저장
    loader = PyPDFLoader(temp_filepath)  # PDF 파일 로드
    pages = loader.load_and_split()  # 페이지별로 분리
    return pages

# 파일이 업로드되면 실행되는 코드
if uploaded_file is not None:
    try:
        # PDF 파일을 로드하여 페이지별로 분리
        pages = pdf_to_document(uploaded_file)

        # 텍스트 분리 설정 (1000자씩 분리하고, 50자의 오버랩 적용)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,  # 각 텍스트 청크의 크기
            chunk_overlap  = 50,  # 청크 간 오버랩 설정
            length_function = len,  # 길이 측정 함수로 len 사용
            is_separator_regex = False,  # 구분자를 정규 표현식으로 사용하지 않음
        )
        texts = text_splitter.split_documents(pages)  # 문서 텍스트 분리

        # 텍스트 임베딩을 위한 OpenAI Embeddings 모델 로드
        embeddings_model = OpenAIEmbeddings()

        # Chroma 데이터베이스가 없으면 새로 생성하고, 있으면 불러오기
        if not os.path.exists(persist_directory):
            chromadb = Chroma.from_documents(
                texts,  # 분리된 텍스트
                embeddings_model,  # 임베딩 모델
                collection_name = 'esg',  # 컬렉션 이름
                persist_directory = persist_directory,  # 저장할 디렉토리 경로
            )
        else:
            chromadb = Chroma(
                persist_directory=persist_directory,  # 기존에 저장된 디렉토리 경로
                embedding_function=embeddings_model,  # 임베딩 모델
                collection_name='esg'  # 컬렉션 이름
            )

        # 질문을 받을 수 있는 섹션 생성
        st.header("PDF에게 질문해보세요!!")
        question = st.text_input('질문을 입력하세요')  # 질문 입력 필드

        # 질문하기 버튼 클릭 시 동작
        if st.button('질문하기'):
            with st.spinner('Wait for it...'):  # 로딩 중 표시
                llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  # OpenAI GPT-3.5 모델 로드
                qa_chain = RetrievalQA.from_chain_type(
                                llm,
                                retriever=chromadb.as_retriever(search_kwargs={"k": 3}),  # 검색에서 최대 3개의 문서 반환
                                return_source_documents=True  # 문서 원본 반환
                            )
                result = qa_chain.invoke({"query": question})  # 질문에 대한 답변 생성
                st.write(result["result"])  # 답변 출력
                
    except Exception as e:
        # 오류 발생 시 오류 메시지 출력
        st.error(f"An error occurred: {e}")