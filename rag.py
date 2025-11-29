import os
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pypdf import PdfReader

# Load PDF and convert to text
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)

# Create vector DB
def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    return vectordb

# Ask questions
def ask_question(query):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory="db", embedding_function=embeddings)

    llm = ChatOpenAI(model="gpt-4o-mini")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    return qa.run(query)

