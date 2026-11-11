from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
from rag.index import add_documents, get_retriever
from agents.workflow import agent_graph, set_retriever
import os

app = FastAPI(title="RAG-Ассистент для студентов")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

initial_retriever = get_retriever() if os.path.exists("rag/vectorstore") else None
set_retriever(initial_retriever)

@app.post("/chat")
async def chat(request: ChatRequest): 
    result = agent_graph.invoke({
        "messages": [("human", request.message)]
    })
    answer = result["messages"][-1].content
    return {"answer": answer}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    os.makedirs("rag/documents", exist_ok=True)
    paths = []
    for file in files:
        path = f"rag/documents/{file.filename}"
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)
        paths.append(path)
    
    new_retriever = add_documents(paths)
    set_retriever(new_retriever)
    return {"status": "ok", "files": len(paths)}