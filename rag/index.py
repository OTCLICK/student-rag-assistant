from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def load_and_split(paths):
    docs = []
    for path in paths:
        try:
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def add_documents(file_paths):
    chunks = load_and_split(file_paths)
    vectorstore_path = "rag/vectorstore"
    if os.path.exists(vectorstore_path):
        db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vectorstore_path)
    return db.as_retriever(search_kwargs={"k": 6})

def get_retriever():
    vectorstore_path = "rag/vectorstore"
    if os.path.exists(vectorstore_path):
        db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        return db.as_retriever(search_kwargs={"k": 6})
    return None
