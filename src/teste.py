import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.llms import BaseLLM
import requests
from dotenv import load_dotenv
load_dotenv('../.env')

import tools
import os


# Define a custom LLM wrapper (adapt for specific LLM, e.g., Google Gemini)
class CustomLLM(BaseLLM):
    api_key=''
    def __init__(self, api_key: str):
        self.api_key = api_key
        # self.model=''
        # self.model = tools.Gemini(
        #     model_name = "gemini-1.5-flash",
        #     apikey = api_key,
        #     system_prompt='You are a helpful assistant',
        # )


    def _generate(self, prompt: str, stop=None):
        return self.model.interact(prompt)
    

    @property
    def _llm_type(self):
        return "custom_llm"

# Streamlit app
st.title("Chat Application with LLM")


# Sidebar settings
st.sidebar.title("Settings")

api_key =  os.environ["GEMINI_KEY"]

# Initialize LLM
llm = CustomLLM(api_key=api_key)

# Memory for chat history
chat_history = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(chat_memory=chat_history, return_messages=True)

# Chat interface
st.subheader("Chat Interface")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display conversation history
if st.session_state["messages"]:
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.text_area("User:", msg["content"], key=f"user_{msg['id']}", height=50, disabled=True)
        elif msg["role"] == "llm":
            st.text_area("LLM:", msg["content"], key=f"llm_{msg['id']}", height=50, disabled=True)

# Input field for user messages
user_input = st.text_input("Your message:")

# Send button to interact with LLM
if st.button("Send"):
    if user_input:
        # Add user message to chat history
        st.session_state["messages"].append({"role": "user", "content": user_input, "id": len(st.session_state["messages"])})
        
        # Get LLM response
        try:
            response = llm._call(user_input)
            st.session_state["messages"].append({"role": "llm", "content": response, "id": len(st.session_state["messages"])})
        except Exception as e:
            st.error(f"Error interacting with the LLM: {e}")
