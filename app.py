# app.py
import streamlit as st

st.set_page_config(layout="wide")

st.title("📄 AI Document Analyzer")
st.write("Upload a PDF document and ask any question about its content.")

# --- UI Components ---
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")
user_question = st.text_input("Ask your question here:")

if st.button("Analyze Document"):
    if uploaded_file is not None and user_question:
        st.write("Analysis logic is coming soon!")
        # Here is where we will call our AI functions in the next steps
    else:
        st.warning("Please upload a PDF and enter a question.")