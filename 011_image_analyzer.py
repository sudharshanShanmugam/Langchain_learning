import streamlit as st
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def analyse_image(image_bytes: bytes, prompt: str) -> str:
    llm = ChatOllama(model="llama3.2-vision")
    image_b64 = encode_image(image_bytes)
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        ]
    )
    response = llm.invoke([message])
    return response.content

st.title("Image Analyser")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    prompt = st.text_area("What would you like to know about this image?", value="Describe this image in detail.")

    if st.button("Analyse"):
        with st.spinner("Analysing..."):
            image_bytes = uploaded_file.read()
            result = analyse_image(image_bytes, prompt)
        st.subheader("Analysis")
        st.write(result)
