import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2")

# prompt_template = PromptTemplate(
#     input_variables=["role", "question"],
#     template= """You are an expert interview coach.
# Your job is to help candidates prepare for {role} interviews.
# When given a question, provide:
# 1. A strong sample answer
# 2. Key points to highlight
# 3. What interviewers are really looking for
# Be concise, practical, and encouraging."""
# )
prompt_template = PromptTemplate(
    input_variables=["role","no_of_line", "question"],
    template="""
You are an expert {role} interview coach.

For the question: {question}, provide a natural, spoken-style answer that the candidate can say directly in an interview.

Requirements:
- Exactly {no_of_line} lines
- Clear, confident, and conversational tone
- No headings, no bullet points, no labels
- Keep it simple and easy to speak aloud
"""
)

st.title("Interview Helper")

role = st.text_area("Applied Role:")
question = st.text_area("Question:")
no_of_line = st.number_input("Number of Lines:", min_value=1, max_value=10)


chains = prompt_template | llm

if question and role:
        response = chains.invoke({"role": role, "question": question, "no_of_line": no_of_line})
        st.write(response.content)


