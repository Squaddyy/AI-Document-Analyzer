# app.py (Final Version with 100% Local AI)
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Core AI Functions (Updated for Local Models) ---

@st.cache_resource
def load_models():
    """Loads all the necessary AI models and caches them."""
    st.write("Loading AI models (this happens only once)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    st.write("✅ AI models loaded successfully.")
    return embedding_model, qa_pipeline

def setup_rag_engine_for_pdf(pdf_text, embedding_model):
    """Takes text, chunks it, and creates embeddings."""
    if not pdf_text: return None, None
    text_chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 900)]
    all_embeddings = embedding_model.encode(text_chunks)
    return text_chunks, all_embeddings

def ask_document(question, text_chunks, all_embeddings, embedding_model, qa_pipeline):
    """Performs the Q&A on the document embeddings using a local model."""
    question_embedding = embedding_model.encode([question])
    scores = np.dot(question_embedding, all_embeddings.T)
    top_k_indices = np.argsort(scores, axis=1)[0][-3:][::-1]

    context = "\n\n".join([text_chunks[i] for i in top_k_indices])

    # Use the local question-answering model
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("📄 AI Document Analyzer (Local AI Version)")
st.write("Upload a PDF and ask a question. This app uses open-source models and runs entirely on its own!")

# Load the AI models
embedding_model, qa_pipeline = load_models()

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
            answer = ask_document(user_question, doc_data['text_chunks'], doc_data['all_embeddings'], embedding_model, qa_pipeline)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please upload a PDF and enter a question.")
