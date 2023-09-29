import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle, os, base64
from dotenv import load_dotenv
import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

st.title('📄 Blattner Tech: Ask My PDF')
st.markdown("<h3 style='text-align: left; color: #01acfa;'>Upload any PDF and get immediate answers to your most pressing questions</h3>", unsafe_allow_html=True)

load_dotenv()

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.image("blattner_tech_logo.png", use_column_width=True)
    st.markdown("FOR INTERNAL USE ONLY")
    
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.info("Please enter your OpenAI API key to continue.")
    
if 'chat_history' not in globals():
  chat_history = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = chat_history

with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([6, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="Ask a question about your document",
        label_visibility="collapsed",
    )
    b.form_submit_button("Send", use_container_width=True)

if "pdf" in globals() and pdf not in ["None", None] and openai_api_key:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text=text)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks,embedding=embeddings)
        
        qa_chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(),
                                                         vectorstore.as_retriever(search_kwargs={'k': 6}),
                                                         return_source_documents=True)
    except:
        st.info("OpenAI API key is invalid. Please enter a valid API key to continue.")

if user_input and "pdf" not in globals():
    st.info("Please upload a document to continue")
elif user_input and openai_api_key:
    try:
        st.info("You Asked: {user_input}")
        with st.spinner("Retrieving Answer..."):
            result = qa_chain({'question': user_input, 'chat_history': st.session_state.chat_history})
            answer = result["answer"]
        st.session_state.chat_history.append((user_input, answer))
        st.info(answer)
    except:
        st.info("Please ask a question about your document to continue")
