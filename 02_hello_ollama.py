

import os 
from langchain_ollama import ChatOllama
# 1. Model (runs locally, no API key needed)
llm = ChatOllama(model="llama3.2")


    
question = input("Question:")
response = llm.invoke(question)
print(response.content)
