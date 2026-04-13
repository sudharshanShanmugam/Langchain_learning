import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever

import streamlit as st

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

@st.cache_resource
def build_rag_chain():
    document = TextLoader("product-data.txt").load()
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
        model="BAAI/bge-base-en-v1.5",
    )
    llm = ChatOpenAI(
        api_key=os.getenv("DEEPINFRA_API_KEY"),
        base_url="https://api.deepinfra.com/v1/openai",
        model="Qwen/Qwen3.5-0.8B",
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)

    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, rephrase it as a standalone question. Do NOT answer it."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant that answers questions using ONLY the context provided below.

Rules you must follow without exception:
1. If the context is empty or does not contain the answer, output exactly: I don't have that information.
2. Do NOT add any explanation, suggestion, or extra sentence after that phrase.
3. Do NOT use your training knowledge. Do NOT mention AI, support teams, or external systems.
4. If the context contains the answer, respond in three sentences or fewer.

Context:
{context}
"""),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)

rag_chain = build_rag_chain()



history = StreamlitChatMessageHistory(key="chat_history")

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer"
)

st.title("Chat with the data in documents")

# Display existing chat history
for msg in history.messages:
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

question = st.chat_input("Your question here:")

if question:
    with st.chat_message("user"):
        st.write(question)

    response = chain_with_history.invoke(
        {"input": question},
        config={"configurable": {"session_id": "default"}}
    )

    with st.chat_message("assistant"):
        st.write(response["answer"])

    with st.expander("Debug: Retrieved context"):
        st.write(response.get("context", "No context retrieved"))