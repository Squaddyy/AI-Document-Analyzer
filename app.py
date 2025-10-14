# app.py (Final Hybrid Version)
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

# --- Core AI Functions (Hybrid Approach) ---

@st.cache_resource
def load_embedding_model():
    """Loads the LOCAL sentence transformer model and caches it."""
    st.write("Loading local AI model for document indexing...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st.write("✅ Indexing model loaded successfully.")
    return model

def setup_rag_engine_for_pdf(pdf_text, embedding_model):
    """Takes text, chunks it, and creates embeddings with the local model."""
    if not pdf_text: return None, None
    text_chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 900)]
    all_embeddings = embedding_model.encode(text_chunks)
    return text_chunks, all_embeddings

def ask_document(question, text_chunks, all_embeddings, embedding_model, generative_model):
    """Performs Q&A using local search and a powerful generative model for the answer."""
    question_embedding = embedding_model.encode([question])
    scores = np.dot(question_embedding, all_embeddings.T)
    top_k_indices = np.argsort(scores, axis=1)[0][-3:][::-1]
    
    context = "\n\n".join([text_chunks[i] for i in top_k_indices])
    
    # BRING BACK: The powerful Gemini prompt for high-quality answers
    prompt = f"""
    You are an expert Q&A assistant. Your task is to answer the user's question based *only* on the provided context from a PDF document.
    Synthesize a concise and accurate answer. If the answer is not in the context, clearly state that the information is not available in the document.

    CONTEXT:
    ---
    {context}
    ---

    QUESTION: {question}

    ANSWER:
    """
    
    response = generative_model.generate_content(prompt)
    return response.text

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("📄 AI Document Analyzer (Pro Version)")
st.write("Upload a PDF and ask a question. This app uses a hybrid AI approach for the best results!")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    # BRING BACK: The Google Generative AI model for Q&A
    generative_model = genai.GenerativeModel('models/gemini-pro-latest')
    # Load the local model for embeddings
    embedding_model = load_embedding_model()
except Exception as e:
    st.error(f"Error initializing AI models: {e}")
    st.stop()

if 'document_data' not in st.session_state:
    st.session_state.document_data = None

uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
user_question = st.text_input("Ask your question here:")

if uploaded_file is not None:
    if st.session_state.document_data is None or st.session_state.document_data.get('file_name') != uploaded_file.name:
        with st.spinner("Reading and indexing the document..."):
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
