import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load API key from .env file
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit App setup
st.set_page_config(page_title="🤖 Advanced Chatbot Interface")
st.title("🤖 AI Chatbot with Memory")

# Sidebar controls
model_name = st.sidebar.selectbox(
    "Select Groq Model",
    [
        "gemma2-9b-it", "deepseek-r1-distill-llama-70b", "llama3-8b-8192",
        "mixtral-8x7b", "command-r-plus", "llama3-70b-8192"
    ]
)

temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.0, 0.9
)

max_token = st.sidebar.slider(
    "Max Token", 100, 400, 200
)

# Initialize memory and history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.chat_input("Write something here...:")

if user_input:
    st.session_state.history.append(("user", user_input))

    #  Instantiate LLM here 
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_token
    )

    # Build conversation chain with memory
    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

    # Get response from LLM
    ai_response = conv.predict(input=user_input)

    # Add assistant response to history
    st.session_state.history.append(("assistant", ai_response))

# Display chat history
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)


