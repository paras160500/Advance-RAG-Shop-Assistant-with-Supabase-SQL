#----------------------------------------------------------------------------------------------
#                                       Import Statements
#----------------------------------------------------------------------------------------------

import os 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Annotated,List,TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone 
from langgraph.graph import StateGraph , END, add_messages
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
from langgraph_functionality import chat
#----------------------------------------------------------------------------------------------
#                                       Init Statements
#----------------------------------------------------------------------------------------------

load_dotenv()
OPENAI_API = os.getenv("OPEN_AI_API")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_KEY)
index_name = "shop-product-sample"
index = pc.Index(index_name)
embedding_model = OpenAIEmbeddings(
    api_key=OPENAI_API,
    model="text-embedding-3-small"
)
vector_store = PineconeVectorStore(
    index = index ,
    embedding= embedding_model,
    text_key="Description"
)
llm = ChatOpenAI(
    api_key=OPENAI_API,
    model = "gpt-4o-mini",
    temperature=0
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Shop Catalog Chatbot")

query = st.text_input("Ask query...")

if st.button("Get Answer"):
    user_id = "user_1"
    if query:
        answer , chat_history = chat(user_id , query)
        st.session_state.chat_history = chat_history
        st.write("Answer :- " , answer)

        with st.expander("Chat History"):
            for msg in st.session_state.chat_history:
                if isinstance(msg, HumanMessage):
                    st.markdown(f"**👤 You:** {msg.content}")
                elif isinstance(msg, AIMessage):
                    st.markdown(f"**🤖 Bot:** {msg.content}")