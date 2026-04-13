import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatOllama(model="llama3.2")

# 1st chain 
topic_prompt = PromptTemplate(
    input_variables=["domain"],
    template="""
You are a content strategist.

Given the domain: {domain}, suggest 5 trending and engaging blog topics.

Return only one best topic.
"""
)

# 2nd chain

blog_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are a professional blog writer.

Write a detailed, engaging blog on the topic: {topic}.

Requirements:
- Catchy introduction
- Clear explanation
- Real-world examples
- Strong conclusion
- Simple and readable tone
"""
)




first_chain = topic_prompt|llm |StrOutputParser()
second_chain = blog_prompt|llm |StrOutputParser()
final_chain = first_chain | second_chain


st.title("AI Blog generator ")

domain = st.text_input("Enter Domain (e.g., AI, Web Development, Finance):")

if st.button("Generate") and domain:
    result = final_chain.invoke({"domain": domain})
    st.write(result)
