# app.py (Final Lightweight Version)
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import google.generativeai as genai
import os

# --- Core AI Functions (Updated for Google's Embedding API) ---

def setup_rag_engine_for_pdf(pdf_text):
    if not pdf_text:
        return None, None

    text_chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 900)]

    # NEW: Create embeddings using the Google API
    response = genai.embed_content(model='models/embedding-001',
                                   content=text_chunks,
                                   task_type="retrieval_document")
    all_embeddings = np.array(response['embedding'])

    return text_chunks, all_embeddings

def ask_document(question, text_chunks, all_embeddings, generative_model):
    # NEW: Create embedding for the question using the Google API
    response = genai.embed_content(model='models/embedding-001',
                                   content=question,
                                   task_type="retrieval_query")
    question_embedding = np.array(response['embedding'])

    scores = np.dot(question_embedding, all_embeddings.T)
    top_k_indices = np.argsort(scores)[-3:][::-1]

    context = "\n\n".join([text_chunks[i] for i in top_k_indices])
    prompt = f"Answer the question based *only* on the context.\n\nCONTEXT:---{context}---\n\nQUESTION: {question}\n\nANSWER:"
    response = generative_model.generate_content(prompt)
    return response.text

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("📄 AI Document Analyzer")
st.write("Upload a PDF document and ask any question about its content.")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    generative_model = genai.GenerativeModel('models/gemini-pro-latest')
except Exception as e:
    st.error(f"Error initializing AI models: {e}")
    st.stop()

if 'document_data' not in st.session_state:
    st.session_state.document_data = None

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
user_question = st.text_input("Ask your question here:")

if uploaded_file is not None:
    if st.session_state.document_data is None or st.session_state.document_data['file_name'] != uploaded_file.name:
        with st.spinner("Reading and indexing the document..."):
            file_bytes = uploaded_file.read()
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            extracted_text = "".join(page.get_text() for page in pdf_document)
            text_chunks, all_embeddings = setup_rag_engine_for_pdf(extracted_text)
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
            answer = ask_document(user_question, doc_data['text_chunks'], doc_data['all_embeddings'], generative_model)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please upload a PDF and enter a question.")
