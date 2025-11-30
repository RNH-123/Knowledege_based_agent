import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain imports
from langchain import HuggingFacePipeline, HuggingFaceEmbeddings, RetrievalQA
from langchain_community.vectorstores import Chroma

# Transformers for local LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -------------------------------
# Setup local LLM
# -------------------------------
model_name = "tiiuae/falcon-7b-instruct"  # HuggingFace free model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create HuggingFace pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------------
# Load PDF and convert to text
# -------------------------------
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# -------------------------------
# Split text into chunks
# -------------------------------
def split_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# -------------------------------
# Create vector database
# -------------------------------
def create_vector_store(chunks, persist_dir="db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Free embedding model
    vectordb = Chroma.from_texts(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

# -------------------------------
# Ask questions from PDF content
# -------------------------------
def ask_question(query, persist_dir="db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever()
    )

    return qa.run(query)

# -------------------------------
# Optional test when running this file directly
# -------------------------------
if __name__ == "__main__":
    pdf_path = "example.pdf"  # Replace with your PDF
    if not os.path.exists(pdf_path):
        print(f"PDF file '{pdf_path}' not found!")
    else:
        text = load_pdf(pdf_path)
        chunks = split_text(text)
        create_vector_store(chunks)
        question = "What is this PDF about?"
        answer = ask_question(question)
        print(f"Q: {question}\nA: {answer}")
