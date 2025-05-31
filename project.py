import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from hyDE import query_output as hyde_answer
from multi_query import query_output as multiquery_answer
from single_query import query_output as singlequery_answer

# Title
st.title("Retrieval Augmented Generation Question Answering App")

# Dropdown to choose method
method = st.selectbox("Choose RAG method", ["Single Query", "Multi-Query", "HyDE"])

# Upload PDF and save
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved '{uploaded_file.name}'")

# List available PDFs
pdf_files = [f for f in os.listdir() if f.endswith(".pdf")]
selected_pdf = st.selectbox("Select a PDF for retrieval", pdf_files)

# preprocess.py
def build_retriever_from_pdf(pdf_path):
    # Load and split the document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    splits = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)

    return vectorstore.as_retriever()

# To avoid repeated retrieval of the same document
@st.cache_resource
def get_retriever(pdf_path):
    return build_retriever_from_pdf(pdf_path)


# Question input
question = st.text_input("Ask a question")

# Submit and get answer
if st.button("Get Answer") and question and selected_pdf:
    retriever = get_retriever(selected_pdf)
    if method == "Single Query":
        answer = singlequery_answer(question, retriever)
    elif method == "Multi-Query":
        answer = multiquery_answer(question, retriever)
    elif method == "HyDE":
        answer = hyde_answer(question, retriever)
    else:
        answer = "Invalid method selected."

    st.text_area("Answer", value=answer, height=200)
