import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2")

# 1st chain → Generate Subject Line
subject_prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template="""
You are a marketing expert.

Create a high-converting email subject line for:
Product/Service: {product}
Target Audience: {audience}

Make it:
- Catchy
- Short
- Click-worthy

Return only one subject line.
"""
)

# 2nd chain → Generate Email Content
email_prompt = PromptTemplate(
    input_variables=["subject"],
    template="""
You are a professional email copywriter.

Write a marketing email based on this subject line:
Subject: {subject}

Requirements:
- Personalized greeting
- Engaging opening
- Highlight benefits (not just features)
- Include a clear Call-To-Action (CTA)
- Friendly and persuasive tone
- Keep it concise and impactful

Format:
Subject:
Email Body:
"""
)

# Chains
first_chain = subject_prompt | llm | StrOutputParser()
second_chain = email_prompt | llm | StrOutputParser()

# Sequential chain
final_chain = first_chain | second_chain

# Streamlit UI
st.title("📧 AI Marketing Email Generator")

product = st.text_input("Enter Product/Service:")
audience = st.text_input("Enter Target Audience:")

if st.button("Generate Email") and product and audience:
    result = final_chain.invoke({
        "product": product,
        "audience": audience
    })
    st.write(result)
else:
    st.write("Please enter both product and audience to generate the email.")