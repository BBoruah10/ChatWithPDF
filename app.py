import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    """""
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


"""
def main():
    # Set the page configuration
    st.set_page_config(page_title="Chat with PDF", layout="wide", initial_sidebar_state="expanded")

    # Custom CSS for fonts and styles
    st.markdown(
        """
        <style>
        /* Global font settings */
        body {
            font-family: 'Arial', sans-serif;
            font-size: 16px;
            line-height: 1.6;
            color: #333;
        }

        /* Header */
        .title {
            font-family: 'Helvetica', sans-serif;
            font-size: 42px;
            font-weight: 700;
            color: #4CAF50;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 30px;
        }

        /* Text input */
        .stTextInput>div>div>input {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
            padding: 12px;
            border-radius: 12px;
            border: 1px solid #4CAF50;
            color: #333;
        }

        /* Sidebar styles */
        .sidebar-title {
            font-family: 'Helvetica', sans-serif;
            font-size: 28px;
            font-weight: 700;
            color: #4CAF50;
            margin-bottom: 15px;
        }

        .sidebar-text {
            font-size: 18px;
            color: #666;
            font-family: 'Arial', sans-serif;
            line-height: 1.5;
        }

        /* File uploader button */
        .stButton>button {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
            background-color: #4CAF50;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Footer */
        .footer {
            font-family: 'Arial', sans-serif;
            font-size: 14px;
            color: #888;
            text-align: center;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Header section with a stylish title
    st.markdown('<div class="title">Chat with PDF using Gemini 💁</div>', unsafe_allow_html=True)

    # User input area with a clean input box
    user_question = st.text_input("Ask a Question from the PDF Files", placeholder="Enter your question here...")

    # Process the question
    if user_question:
        user_input(user_question)

    # Sidebar with improved layout and font size
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Menu:</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-text">Upload your PDF files and click on the Submit & Process button.</div>', unsafe_allow_html=True)

        # File uploader with a more modern look
        pdf_docs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

        # Submit button with custom font and style
        if st.button("Submit & Process", use_container_width=True):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done ✅")

    # Custom footer with smaller font and a polished appearance
    st.markdown('<div class="footer">Powered by Gemini PDF Chatbot | 2024</div>', unsafe_allow_html=True)
if __name__ == "__main__":
    main()