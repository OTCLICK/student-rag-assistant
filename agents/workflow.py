from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_gigachat import GigaChat
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import os
from dotenv import load_dotenv
import pyttsx3  

load_dotenv()

# Глобальный retriever
current_retriever = None

def set_retriever(retriever):
    global current_retriever
    current_retriever = retriever

llm = GigaChat(
    credentials=os.getenv("GIGACHAT_TOKEN"),
    model="GigaChat-Max",
    temperature=0.3,
    verify_ssl_certs=False
)

# TTS engine с поддержкой русского
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 170)  # скорость речи

# Поиск русского голоса
voices = tts_engine.getProperty('voices')
russian_voice = None
for voice in voices:
    if "ru" in voice.id or "Russian" in voice.name:
        russian_voice = voice.id
        break

if russian_voice:
    tts_engine.setProperty('voice', russian_voice)
else:
    print("Русский голос не найден. Используется системный голос по умолчанию.")

def speak(text: str):
    tts_engine.say(text)
    tts_engine.runAndWait()

# Состояние агента
class AgentState(TypedDict):
    messages: Annotated[list, "messages"]
    use_rag: bool

# RAG tool
@tool
def rag_search(query: str) -> str:
    """Выполняет семантический поиск по загруженным PDF-документам студента."""
    global current_retriever
    if not current_retriever:
        return "База знаний пуста. Загрузите PDF с лекциями."

    docs = current_retriever.invoke(query)
    if not docs:
        return "По вашему запросу ничего не найдено в загруженных документах."

    result = ["**Найдено в ваших лекциях:**\n"]
    for i, doc in enumerate(docs[:5], 1):
        text = doc.page_content.strip()
        source = doc.metadata.get("source", "неизвестно")
        page = doc.metadata.get("page", 0) + 1
        result.append(f"**Источник {i}:** {source}, стр. {page}")
        result.append(f"> {text}\n")
    return "\n".join(result)

llm_with_tools = llm.bind_tools([rag_search])

def _extract_text(message) -> str:
    if isinstance(message, (AIMessage, HumanMessage)):
        return message.content
    if isinstance(message, tuple):
        return message[1]
    return str(message)

# PlannerAgent
def planner(state: AgentState):
    last_message = state["messages"][-1]
    last_user_text = _extract_text(last_message).lower()
    rag_keywords = ["лекци", "конспект", "pdf", "документ"]
    use_rag = any(keyword in last_user_text for keyword in rag_keywords)

    plan_text = (
        "План действий:\n"
        f"1. Тип запроса: {'учебный' if use_rag else 'общий'}.\n"
        f"2. {'Выполнить поиск по лекциям.' if use_rag else 'Ответить без обращения к лекциям.'}\n"
        "3. Сформировать финальный ответ."
    )
    plan_message = AIMessage(content=plan_text)
    return {
        "messages": state["messages"] + [plan_message],
        "use_rag": use_rag
    }

# RAGAgent
def rag_agent(state: AgentState):
    last_message = state["messages"][-1]
    query = _extract_text(last_message)
    rag_result = rag_search.invoke({"query": query})
    return {
        "messages": state["messages"] + [AIMessage(content=rag_result)],
        "use_rag": state.get("use_rag", False)
    }

# Chatbot
def chatbot(state: AgentState):
    if state.get("use_rag"):
        response = llm_with_tools.invoke(state["messages"])
    else:
        response = llm.invoke(state["messages"])
    return {
        "messages": state["messages"] + [response],
        "use_rag": state.get("use_rag", False)
    }

# AnalysisAgent с TTS
def analysis_agent(state: AgentState):
    last_message = state["messages"][-1]
    text = _extract_text(last_message)

    structured_text = f"--- Финальный ответ:\n{text}\n"
    if state.get("use_rag"):
        structured_text += "\n(Данные получены из ваших лекций)"

    if state.get("tts_enabled", True):
        speak(structured_text)

    return {
        "messages": state["messages"] + [AIMessage(content=structured_text)],
        "use_rag": state.get("use_rag", False)
    }

graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("rag_agent", rag_agent)
graph.add_node("chatbot", chatbot)
graph.add_node("analysis_agent", analysis_agent)
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", lambda s: "rag_agent" if s.get("use_rag") else "chatbot")
graph.add_edge("rag_agent", "chatbot")
graph.add_edge("chatbot", "analysis_agent")
agent_graph = graph.compile()
