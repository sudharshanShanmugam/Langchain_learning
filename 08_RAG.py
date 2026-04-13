from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain



document = TextLoader("product-data.txt").load()
embeddings = OllamaEmbeddings(model="llama3.2")

llm = ChatOllama(model="llama3.2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
chunk_overlap=200)

chunks=text_splitter.split_documents(document)

vector_store =Chroma.from_documents (chunks, embeddings)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

prompt_template = ChatPromptTemplate.from_messages([
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


qa_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, qa_chain)


print("chat with the data in documents")
question = input("your question here: ")

if question:
    response = rag_chain.invoke({"input":question})
    print(response["answer"])