import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai",
)

# Streamlit UI
st.title("chat with memory ")

if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "You are a marketing expert."}]

user_input = st.text_input("Enter your message:")

if st.button("Generate"):
    st.session_state.history.append({"role": "user", "content": user_input})

    chat_completion = openai.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=st.session_state.history,
    )

    response = chat_completion.choices[0].message.content
    st.session_state.history.append({"role": "assistant", "content": response})
    st.write(response)

# Display chat history
st.write("HISTORY")
for msg in st.session_state.history[1:]:  # skip system message
    st.write(f"**{msg['role'].capitalize()}:** {msg['content']}")