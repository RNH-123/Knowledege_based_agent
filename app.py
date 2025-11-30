import streamlit as st
import os
from rag import load_pdf, split_text, create_vector_store, ask_question

st.set_page_config(page_title="📚 PDF Q&A", page_icon="📄")
st.title("📚 Knowledge-based PDF Q&A App")
st.write("Upload a PDF and ask questions from it!")

# -------------------------------
# Upload PDF
# -------------------------------
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_file.read())
    st.success("PDF uploaded successfully!")

    text = load_pdf("uploaded.pdf")

    if st.button("Create Knowledge Base"):
        st.write("Splitting text...")
        chunks = split_text(text)

        st.write("Creating vector store...")
        create_vector_store(chunks)

        st.success("Knowledge base created!")

# -------------------------------
# Ask a question
# -------------------------------
query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        answer = ask_question(query)
        st.write("### Answer:")
        st.write(answer)
