import os 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Annotated,List,TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone 
from langgraph.graph import StateGraph , END, add_messages
from langgraph.checkpoint.memory import MemorySaver

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

class ChatState(TypedDict):
    messages : Annotated[List[BaseMessage] , add_messages]
    question : str 
    context : str 
    answer : str 
    need_retrieval : bool 


def history_agent(state : ChatState):
    history = state['messages']
    question = state['question']

    if len(history) == 0:
        return {
            "answer" : "",
            "need_retrieval" : True
        }
    
    history_text = ""

    for msg in history:
        if isinstance(msg , HumanMessage):
            history_text += f"User : {msg.content}"
        elif isinstance(msg,AIMessage):
            history_text += f"Assistant : {msg.content}"

    prompt = f"""
    You are a conversation memory system.
    Previous Conversation:
    {history_text}
    New user Question:
    {question}
    
    INSTRUCTION:
    - If previous conversation contains enough information, answer user directly
    - If previous conversation doesnot contain the answer reply exactly:
        NOT_FOUND

    Answer:
    """

    response = llm.invoke(prompt)
    result = response.content.strip()

    if result == "NOT_FOUND":
        return {
            "answer" : "",
            "need_retrieval" : True
        }
    return {
        "answer" : result,
        "need_retrieval" : False
    }


def route_question(state : ChatState):
    if state['need_retrieval']:
        return "pinecone"
    else:
        return "save"
    

def pinecone_search(state : ChatState):
    docs = vector_store.similarity_search(state['question'] , k = 3)
    context = "" 
    for doc in docs:
        context += f"""
        Product:
        {doc.page_content}

        Information:
        {doc.metadata}
        """
        
    return {
        "context" : context
    }


def generate_from_context(state : ChatState):
    prompt = f""" 
    You are a shopping assistant
    Use this product information:
    {state['context']}

    Question : {state['question']}

    Answer Clearly
    """ 
    response = llm.invoke(prompt)

    return {
        "answer" : response.content
    }



def save_memory(state : ChatState):
    user_message=f"""

    Question:

    {state["question"]}


    Retrieved Context:

    {state["context"]}

    """

    state['messages'].append(HumanMessage(content = user_message))
    state['messages'].append(AIMessage(content = state['answer']))

    return state 


graph = StateGraph(ChatState)
graph.add_node("history" , history_agent)
graph.add_node("generate" , generate_from_context)
graph.add_node("pinecone" , pinecone_search)
graph.add_node("save" , save_memory)
graph.set_entry_point("history")
graph.add_conditional_edges(
    "history" , 
    route_question ,
    {
        "pinecone" : "pinecone",
        "save" : "save"
    }
)
graph.add_edge("pinecone" , "generate")
graph.add_edge("generate" , "save")
graph.add_edge("save", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

def chat(user_id , question):
    result = app.invoke({
        "messages" : [],
        "question" : question,
        "context" : "",
        "answer" : "",
        "need_retrieval" : False, 
    }, config={"configurable" : {"thread_id" : user_id}})

    return result['answer'] , result['messages']


# user_id = "paras"

# while True:
#     query = input("Enter query :- ")
#     if query == "exit":
#         break 
#     answer , chat_history = chat(user_id , query)
#     print("\nYou :- " , query)
#     print("\nAssistant :- " , answer)
#     print("\n\n--------------------------------------------------------------")
#     print(chat_history)
#     print("\n\n--------------------------------------------------------------")