import streamlit as st
from rag import load_pdf, split_text, create_vector_store, ask_question

st.title("📚 Knowledge Base Agent (Beginner Friendly)")

menu = st.sidebar.selectbox("Menu", ["Upload Documents", "Ask Questions"])

if menu == "Upload Documents":
    st.subheader("Upload your PDF files")
    pdf = st.file_uploader("Choose PDF", type="pdf")

    if pdf:
        text = load_pdf(pdf)
        chunks = split_text(text)
        create_vector_store(chunks)
        st.success("Document added to knowledge base!")

elif menu == "Ask Questions":
    st.subheader("Ask a question about your documents")

    query = st.text_input("Enter your question")

    if st.button("Ask"):
        answer = ask_question(query)
        st.write("### Answer:")
        st.write(answer)
