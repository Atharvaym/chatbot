import streamlit as st  
import torch
import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA  

# Set up Groq API Key (Replace with your actual key)
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-70b-8192"  # Latest Groq-supported Llama 3 model

# Streamlit UI
st.title("🧠 RAG Chat")
st.write("📄 Upload a document and ask questions!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a PDF document", type=["pdf"])
if uploaded_file is not None:
    doc_path = f"temp_{uploaded_file.name}"
    with open(doc_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF document
    loader = PyPDFLoader(doc_path)
    documents = loader.load()

    # Generate embeddings & store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    st.success("✅ Embeddings stored in FAISS.")

    # Function to call Groq API for text generation
    def call_groq_api(prompt):
        url = "https://api.groq.com/openai/v1/chat/completions"  # Corrected endpoint
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": GROQ_MODEL,
            "messages": [{"role": "system", "content": "You are a helpful AI answering questions about REST vs GraphQL."},
                         {"role": "user", "content": prompt}],
            "max_tokens": 500
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ Error: {response.json()}"

    # Ask a question
    query = st.text_input("🔍 Ask a question about REST vs GraphQL:")
    if query:
        retrieved_docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = call_groq_api(prompt)

        st.write("💬 **Answer:**", response)

    # Clean up temp file
    os.remove(doc_path)
