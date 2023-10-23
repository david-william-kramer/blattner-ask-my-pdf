import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle, os, base64
from dotenv import load_dotenv
import sys
import openai
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, TransformChain, SimpleSequentialChain

os.environ['WOLFRAM_ALPHA_APPID'] = 'Y8XVGA-EAQWARHA6U'
os.environ["SERPER_API_KEY"] = '099fb81a3b2ba61ebdbe08e6424d3bb44bdd0505'

st.title('📄 Merlin Cyber: Ask My Tax PDF')
st.markdown("<h3 style='text-align: left; color: #0076fc;'>Upload any tax file and get immediate answers to your most pressing questions</h3>", unsafe_allow_html=True)

load_dotenv()

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password") #Replace with user_input box eventually
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    st.image("blattner_tech_logo.png", use_column_width=True)
    st.image("Merlin-Cyber.png", use_column_width = True)
    st.markdown("<h4 style='text-align: left; color: #0076fc;'>(For Internal Use Only)</h4>", unsafe_allow_html=True)
    
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key
else:
    st.info("Please enter your OpenAI API key to continue.")

@st.cache_data(show_spinner = "Loading Retrieval Augmented Computation (RAC) Agent...")
def load_computation_agent():
  if openai_api_key:
    llm = OpenAI(temperature=0)
    chatopenai = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0.2)
    wolfram = WolframAlphaAPIWrapper()
    
    tools = [Tool(
            name="Wolfram",
            func=wolfram.run,
            description="Useful for when you need to answer questions requiring computation or in topic areas like math, science, geography, etc.")]
    
    mrkl = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    return llm, chatopenai, mrkl

if openai_api_key:
    llm, chatopenai, mrkl = load_computation_agent()
    
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

@st.cache_data(show_spinner = "Passing document embeddings to vector database...")
def load_document(pdf):
  if "pdf" in globals() and pdf not in ["None", None] and openai_api_key:
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
      return qa_chain

if "pdf" in globals() and pdf not in ["None", None] and openai_api_key:
    qa_chain = load_document(pdf)

computation_classifier_prompt = """
I want you to act as an automated agent with experience in the field of tax preparation.
You will take, as input, a user request for tax information, as well as relevant text excerpts from a tax form.
Based on the question and corresponding information from the form, please decide whether or not computation is required to satisfy the request.
Only return "Yes" or "No".

User Request: {text}
Tax Form Context: {context}
Computation Decision:
"""

if openai_api_key:
    computation_decision_template = PromptTemplate(template=computation_classifier_prompt, input_variables=["text", "context"])
    computation_decision_chain = LLMChain(llm=chatopenai, prompt=computation_decision_template)

if user_input and "pdf" not in globals():
    st.info("Please upload a document to continue")
elif user_input and openai_api_key:
    #try:
    st.info(f"You Asked: {user_input}")
    with st.spinner("Retrieving Answer..."):
        result = qa_chain({'question': user_input, 'chat_history': st.session_state.chat_history})
        answer = result["answer"]
    st.session_state.chat_history.append((user_input, answer))
    with st.spinner("Building Chain-of-Thought..."):
        llm_response = computation_decision_chain.run({"text": user_input, "context": context})

    if llm_response.strip().lower() == "yes":
      with st.spinner("Applying self-consistency..."):
        full_contextualized_question = "Solve this problem through step-by-step computation given the tax form context below:\n\nUser Request: {text}\n\nTax Form Information: {context}. Please show your reasoning step-by-step, and make sure you end with a line that starts with 'Therefore'."
        final_computation_prompt = PromptTemplate(template=full_contextualized_question, input_variables=["text", "context"])
        computation_chain = LLMChain(llm=chatopenai, prompt=final_computation_prompt)
        eligible_final_answers_dict = {}
        for i in range(0, 3):
          response = computation_chain.run({"text": user_input, "context": context})
          split_response = response.split("\n")
          final_answer_response_line = [sentence for sentence in split_response if sentence.startswith("Therefore")]
          if len(final_answer_response_line) > 0:
            final_answer_words = final_answer_response_line[-1].split()
            final_answer = final_answer_words[-1].strip(".")
            if final_answer.replace("$", "").replace(",", "").replace(".", "").isdigit():
              eligible_final_answers_dict[final_answer] = final_answer_response_line[-1]
        if len(eligible_final_answers_dict) > 1:
          self_consistent_answer = statistics.mode(eligible_final_answers_dict.keys)
          self_consistent_full_answer = eligible_final_answers_dict[self_consistent_answer]
        elif len(eligible_final_answers_dict) > 0:
          self_consistent_answer = list(eligible_final_answers_dict.values())[0]
        else:
          self_consistent_answer = final_answer_response_line
        answer = f"{context}\n\n{self_consistent_answer}"
        st.info(answer)
    else:
      st.info(context)
        
    # except:
    #     st.info("Please ask a question about your document to continue")
