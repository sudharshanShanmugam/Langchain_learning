from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Model
llm = ChatOpenAI(model="gpt-4o-mini",)

# 2. Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])

# 3. Output parser
output_parser = StrOutputParser()

# 4. Chain: prompt | model | parser
chain = prompt | llm | output_parser

# 5. Run it
response = chain.invoke({"input": "What is LangChain in one sentence?"})
print(response)
