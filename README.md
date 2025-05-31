# Retrieval-Augmented-Generation-Question-Answering-App

This is a Retrieval Augmented Generation (RAG) Question Answering App that allows you to upload multiple PDFs and ask questions based on their content.

The app supports three query translation approaches:

Single Query

Multi-Query

HyDE (Hypothetical Document Embeddings)

It uses the phi-4-mini language model and is built with Streamlit to run on a local server.

🚀 Features
Upload and parse multiple PDF documents

Query the documents using advanced RAG techniques

Uses phi-4-mini for language modeling via Ollama

Interactive frontend using Streamlit

🛠️ Installation & Setup
Follow the steps below to run the app locally:

Install Ollama
Download and install Ollama. Then run:

ollama run phi4-mini
ollama pull all-minilm

Install Python dependencies:
pip install -r requirements.txt


Run the Streamlit app

python -m streamlit run project.py

💡 Query Translation Approaches

Single Query: Directly queries the uploaded PDFs.
Multi-Query: Generates multiple rephrased versions of the question for broader retrieval.
HyDE: Uses hypothetical document generation for better context embedding.

🧠 Model Details
Language Model: phi-4-mini (via Ollama)

Embedding Model: all-minilm (via Ollama)

📄 License
MIT License

