__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from streamlit_extras.buy_me_a_coffee import button
from langchain_core.callbacks.base import BaseCallbackHandler
#from dotenv import load_dotenv
#load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")

#OpenAI 키 입력받기
openai_key = st.text.input('OPEN_AI_API_KEY', type="password")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요.", type=["pdf"])
st.write("---")

#Buy me a coffee
button(username="ahekrltql", floating=True, width=221)

def pdf_to_document(uploaded_file):
    # 업로드된 파일을 임시 파일로 저장 후 Document 리스트로 로드
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    docs = loader.load()  # load_and_split 말고 load만 쓰고 밑에서 split
    return docs

# 업로드된 파일이 없으면 아래 코드 실행 중단
if uploaded_file is None:
    st.info("왼쪽에서 PDF 파일을 먼저 업로드해주세요.")
    st.stop()

# 업로드된 파일 처리
docs = pdf_to_document(uploaded_file)

# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(docs)

# Embedding
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=openai_key
)

# Chroma DB
db = Chroma.from_documents(texts, embeddings_model)

# 스트리밍을 처리할 Handler 생성
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text)

# User Input
st.header("PDF에게 질문해보세요!")
question = st.text_input("질문을 입력하세요.")

if st.button("질문하기") and question:
    with st.spinner("Wait for it..."):
        # Retriever
        llm = ChatOpenAI(temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(), llm=llm
        )

        # Prompt Template
        prompt = hub.pull("rlm/rag-prompt")

        # Generate
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)
        generate_llm = ChatOpenAI(model="gpt-4o-mini",temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        rag_chain = (
            {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
            | prompt
            | generate_llm
            | StrOutputParser()
        )

        # Question – 하드코딩 대신 사용자가 입력한 question 사용
        result = rag_chain.invoke(question)
        st.write(result)
