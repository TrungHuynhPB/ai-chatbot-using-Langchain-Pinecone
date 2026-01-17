import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from utils import find_match, query_refiner, get_conversation_string

# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
st.title("Personal AI Bot")

if "responses" not in st.session_state:
    st.session_state.responses = ["How can I assist you?"]

if "requests" not in st.session_state:
    st.session_state.requests = []

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# --------------------------------------------------
# Prompt
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using the provided context. If unknown, say 'I don't know'."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

# --------------------------------------------------
# Memory (v1.0 way)
# --------------------------------------------------
def get_history(session_id: str):
    if session_id not in st.session_state:
        st.session_state[session_id] = InMemoryChatMessageHistory()
    return st.session_state[session_id]

chat = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --------------------------------------------------
# UI
# --------------------------------------------------
query = st.text_input("Query:")

if query:
    with st.spinner("typing..."):
        convo = get_conversation_string()
        refined = query_refiner(convo, query)
        context = find_match(refined)

        response = chat.invoke(
            {"input": f"Context:\n{context}\n\nQuery:\n{query}"},
            config={"configurable": {"session_id": "default"}},
        ).content

    st.session_state.requests.append(query)
    st.session_state.responses.append(response)

# --------------------------------------------------
# History
# --------------------------------------------------
for i, resp in enumerate(st.session_state.responses):
    message(resp, key=str(i))
    if i < len(st.session_state.requests):
        message(
            st.session_state.requests[i],
            is_user=True,
            key=f"{i}_user",
        )
