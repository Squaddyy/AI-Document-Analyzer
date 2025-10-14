# app.py
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def setup_rag_engine_for_pdf(pdf_text, embedding_model):
    if not pdf_text:
        return None, None
    text_chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 900)]
    all_embeddings = embedding_model.encode(text_chunks)
    return text_chunks, all_embeddings

def ask_document(question, text_chunks, all_embeddings, embedding_model, generative_model):
    question_embedding = embedding_model.encode([question])
    scores = np.dot(question_embedding, all_embeddings.T)
    top_k_indices = np.argsort(scores, axis=1)[0][-3:][::-1]
    context = "\n\n".join([text_chunks[i] for i in top_k_indices])
    prompt = f"""
    Answer the user's question based *only* on the provided context from a PDF document.
    If the answer is not in the context, clearly state that the information is not available in the document.
    CONTEXT:---{context}---
    QUESTION: {question}
    ANSWER:
    """
    response = generative_model.generate_content(prompt)
    return response.text

st.set_page_config(layout="wide")
st.title("📄 AI Document Analyzer")
st.write("Upload a PDF document and ask any question about its content.")

try:
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        api_key = st.secrets.get("GEMINI_API_KEY") # Use st.secrets for deployment
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it locally or in Streamlit secrets.")
        st.stop()
    genai.configure(api_key=api_key)
    generative_model = genai.GenerativeModel('models/gemini-pro-latest')
    embedding_model = load_embedding_model()
except Exception as e:
    st.error(f"Error initializing AI models: {e}")
    st.stop()

if 'document_data' not in st.session_state:
    st.session_state.document_data = None

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
user_question = st.text_input("Ask your question here:")

if uploaded_file is not None:
    if st.session_state.document_data is None or st.session_state.document_data['file_name'] != uploaded_file.name:
        with st.spinner("Reading and indexing the document... This happens only once per file."):
            file_bytes = uploaded_file.read()
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            extracted_text = "".join(page.get_text() for page in pdf_document)
            text_chunks, all_embeddings = setup_rag_engine_for_pdf(extracted_text, embedding_model)
            st.session_state.document_data = {
                'file_name': uploaded_file.name,
                'text_chunks': text_chunks,
                'all_embeddings': all_embeddings
            }
        st.success(f"Document '{uploaded_file.name}' is indexed and ready!")

if st.button("Analyze Document"):
    if st.session_state.document_data is not None and user_question:
        with st.spinner("Finding the answer..."):
            doc_data = st.session_state.document_data
            answer = ask_document(user_question, doc_data['text_chunks'], doc_data['all_embeddings'], embedding_model, generative_model)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please upload a PDF and enter a question.")
