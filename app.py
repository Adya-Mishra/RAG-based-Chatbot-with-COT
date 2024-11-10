# app.py

import streamlit as st
from rag_module import generate_response_with_cot

# Set custom page layout and styling
st.set_page_config(page_title="RAG-based Chatbot with COT", page_icon="🤖", layout="centered")


st.markdown("""
    <style>
        .title-style {
            color: #F9F9F9;
            font-size: 30px;
            font-weight: bold;
            text-align: center;
        }
        .input-box {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 10px;
            font-size: 18px;
        }
        .response-box {
            background-color: #323232;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Display a custom title
st.markdown('<p class="title-style">🤖 RAG-based Chatbot with COT</p>', unsafe_allow_html=True)

# Take user query as input with a custom-styled input box
user_query = st.text_input("Ask a question:", placeholder="Type your question here...", key="user_input")

# Add a button to submit the question
if st.button("Get Response"):
    if user_query:
        # Generate response for the query
        response = generate_response_with_cot(user_query)
        
        # Display the chatbot's response with styling
        st.markdown('<div class="response-box"><strong>Chatbot:</strong> ' + response + '</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a question to get a response.")

