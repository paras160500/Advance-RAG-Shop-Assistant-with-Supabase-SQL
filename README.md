# 🛍️ Smart Shop Assistant — Agentic RAG with Supabase SQL & Pinecone

<div align="center">

![Architecture Diagram](diagram.png)

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-blueviolet?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_Store-000000?logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?logo=supabase&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)

</div>

---

## 🧠 What Is This?

**Smart Shop Assistant** is an agentic RAG-powered chatbot for a product catalogue. Ask it anything about products — price, brand, colour, gender, description — and it answers from your actual product data. But it's not just a simple retrieval system.

The key innovation is a **history-aware LangGraph agent** that checks your conversation history *before* going to the vector store. If your previous messages already contain the answer, it replies instantly from memory — no embedding call, no Pinecone query, no LLM generation round-trip. Only when history doesn't have the answer does it hit Pinecone, retrieve the top-3 products, generate a fresh response, and commit everything to memory for next time.

The product data lives in two places simultaneously: **Supabase PostgreSQL** (structured, queryable SQL) and **Pinecone** (semantic vector search) — kept in sync by a dedicated `sync_pinecone.py` script.

---

## ✨ Features

| | |
|---|---|
| 🧠 **History-First Agent** | Checks conversation memory before retrieval — avoids unnecessary Pinecone queries |
| 🔀 **Smart Routing** | `need_retrieval` flag drives the conditional edge — `history → pinecone` or `history → save` |
| 🗄️ **Dual Data Store** | Products in Supabase PostgreSQL + Pinecone vector index, synced by `sync_pinecone.py` |
| 🔍 **Semantic Product Search** | `text-embedding-3-small` embeds combined product text for rich similarity matching |
| 🤖 **GPT-4o-mini Generation** | `temperature=0` — factual, consistent product answers |
| 💬 **Persistent Session Memory** | `MemorySaver` checkpointer per `user_id` — remembers the full conversation thread |
| 📖 **Source Citations** | Answer includes product metadata: name, brand, price, colour, gender |
| 🖥️ **Streamlit Chat UI** | Clean question-answer interface with expandable chat history |

---

## 🏗️ Agent Graph

```mermaid
flowchart TD
    S(["👤 User Query"])

    HIST["🧠 history_agent<br/>Check conversation memory<br/>gpt-4o-mini · temp=0"]

    PINE["🌲 pinecone_search<br/>Similarity search · top_k=3<br/>text-embedding-3-small"]

    GEN["🤖 generate_from_context<br/>Shopping assistant prompt<br/>gpt-4o-mini · temp=0"]

    SAVE["💾 save_memory<br/>Append Q + A to messages<br/>MemorySaver per user_id"]

    E(["✅ Answer Returned"])

    S --> HIST
    HIST -->|"NOT_FOUND in history"| PINE
    HIST -->|"Found in history"| SAVE
    PINE --> GEN --> SAVE --> E

    classDef entry   fill:#7c3aed,stroke:#5b21b6,color:#fff
    classDef hist    fill:#f59e0b,stroke:#d97706,color:#fff
    classDef search  fill:#10b981,stroke:#059669,color:#fff
    classDef gen     fill:#3b82f6,stroke:#2563eb,color:#fff
    classDef mem     fill:#6366f1,stroke:#4f46e5,color:#fff
    classDef done    fill:#059669,stroke:#047857,color:#fff

    class S entry
    class HIST hist
    class PINE search
    class GEN gen
    class SAVE mem
    class E done
```

> **The smart shortcut:** if the user asks a follow-up question already answered in the conversation (`HIST` finds it), the agent jumps straight to `save_memory` → `END` — skipping Pinecone and the LLM generation entirely. Faster and cheaper.

---

## 🗂️ Project Structure

```
Advance-RAG-Shop-Assistant-with-Supabase-SQL/
│
├── app.py                      # Streamlit UI — chat interface + session history
├── langgraph_functionality.py  # LangGraph agent: state, nodes, graph, chat()
├── langgraph_sample.ipynb      # Interactive notebook walkthrough of the full agent
│
├── sync_pinecone.py            # Sync Supabase products → Pinecone (batch upsert)
├── insertion_data.py           # Load data.csv → Supabase products table
├── table_cration.py            # Create the products table in Supabase
├── sample_table_creation_query.txt  # Raw SQL for the products table
│
├── data.csv                    # ~130 fashion products (Nike, Adidas, Vans, etc.)
├── requirements.txt            # All dependencies (pinned)
├── pyproject.toml              # Project metadata
└── main.py                     # Root entry point
```

---

## 🗄️ Data Pipeline — Supabase → Pinecone

The product data flows through three setup scripts before the chatbot can answer any questions:

### Step 1 — Create the Table (`table_cration.py`)

```python
CREATE TABLE IF NOT EXISTS products (
    ProductID    INT PRIMARY KEY,
    ProductName  VARCHAR(255),
    ProductBrand VARCHAR(255),
    Gender       VARCHAR(50),
    Price        VARCHAR(50),
    Description  TEXT,
    PrimaryColor VARCHAR(50)
);
```

Connects to Supabase PostgreSQL via `psycopg2` using `DATABASE_URL` with `sslmode="require"`.

### Step 2 — Insert Data (`insertion_data.py`)

Reads `data.csv` (≈ 130 fashion products — Nike, Adidas, Vans, and more) and inserts each row into the `products` table. Checks if data already exists before inserting to prevent duplicates.

### Step 3 — Sync to Pinecone (`sync_pinecone.py`)

Fetches all rows from Supabase, builds a rich combined text string per product for high-quality embeddings, and upserts in batches of 100:

```python
texts = [
    f"{row['description']} {row['productname']} {row['productbrand']} "
    f"{row['gender']} {row['price']} {row['primarycolor']}"
    for _, row in batch.iterrows()
]

metadata = [{
    'ProductName':  row['productname'],
    'ProductBrand': row['productbrand'],
    'Gender':       row['gender'],
    'Price':        row['price'],
    'PrimaryColor': row['primarycolor'],
    'Description':  row['description']
}]

index.upsert(vectors=zip(ids, embeds, metadata))
```

> After this, the Pinecone index `shop-product-sample` (1536-dim, dotproduct, AWS Serverless) is fully loaded and ready for semantic search.

---

## 🤖 How the Agent Works

### State Definition

```python
class ChatState(TypedDict):
    messages:       Annotated[List[BaseMessage], add_messages]  # full conversation thread
    question:       str        # current user question
    context:        str        # retrieved product info from Pinecone
    answer:         str        # generated response
    need_retrieval: bool       # routing flag
```

### Node 1 — `history_agent`

Checks if the current question can be answered from previous conversation messages. If history is empty, sets `need_retrieval=True` immediately. Otherwise, sends history + question to the LLM with a strict instruction:

```python
prompt = f"""
You are a conversation memory system.
Previous Conversation: {history_text}
New user Question: {question}

INSTRUCTION:
- If previous conversation contains enough information, answer user directly
- If previous conversation does not contain the answer reply exactly: NOT_FOUND
"""
```

If the LLM responds with `NOT_FOUND` → `need_retrieval = True` → route to Pinecone.
If it responds with an answer → `need_retrieval = False` → route straight to `save_memory`.

### Node 2 — `pinecone_search`

```python
docs = vector_store.similarity_search(state['question'], k=3)
context = ""
for doc in docs:
    context += f"Product: {doc.page_content}\nInformation: {doc.metadata}"
```

Returns the top-3 most semantically similar products with full metadata.

### Node 3 — `generate_from_context`

```python
prompt = """
You are a shopping assistant.
Use this product information: {context}
Question: {question}
Answer Clearly.
"""
```

Generates a clear, product-specific answer from the retrieved context.

### Node 4 — `save_memory`

Appends the question (+ retrieved context) as a `HumanMessage` and the answer as an `AIMessage` to the conversation thread. The `MemorySaver` checkpointer persists this per `thread_id` (= `user_id`).

### Graph Wiring

```python
graph.set_entry_point("history")
graph.add_conditional_edges("history", route_question, {
    "pinecone": "pinecone",   # need_retrieval = True
    "save":     "save"        # need_retrieval = False (answered from memory)
})
graph.add_edge("pinecone", "generate")
graph.add_edge("generate", "save")
graph.add_edge("save", END)

app = graph.compile(checkpointer=MemorySaver())
```

---

## 📓 Interactive Notebook — `langgraph_sample.ipynb`

`langgraph_sample.ipynb` is a step-by-step walkthrough of the entire agent — ideal for understanding or experimenting with the graph before running the full Streamlit app.

It walks through:
- Setting up Pinecone, OpenAI embeddings, and `PineconeVectorStore`
- Defining `ChatState` with annotated message history
- Building each node (`history_agent`, `pinecone_search`, `generate_from_context`, `save_memory`)
- Wiring the conditional graph and compiling with `MemorySaver`
- Running a live interactive loop in the notebook:

```python
user_id = "paras"
while True:
    query = input("Enter query :- ")
    if query == "exit":
        break
    answer, chat_history = chat(user_id, query)
    print("\nYou :- ", query)
    print("\nAssistant :- ", answer)
```

---

## 📦 Installation & Local Setup

### Prerequisites

- Python 3.12+
- [Supabase](https://supabase.com) project (free tier) with PostgreSQL enabled
- [Pinecone](https://pinecone.io) account (free tier)
- [OpenAI](https://platform.openai.com) API key

### 1. Clone

```bash
git clone https://github.com/paras160500/Advance-RAG-Shop-Assistant-with-Supabase-SQL.git
cd Advance-RAG-Shop-Assistant-with-Supabase-SQL
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

```env
# Supabase PostgreSQL
DATABASE_URL=postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key

# OpenAI
OPEN_AI_API=your_openai_api_key
```

### 4. Set up the data pipeline (one-time)

```bash
# 1 — Create the products table in Supabase
python table_cration.py

# 2 — Insert product data from data.csv
python insertion_data.py

# 3 — Embed and sync all products into Pinecone
python sync_pinecone.py
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 💬 Example Conversations

```
You: Show me some Nike shoes for women
Bot: Here are some Nike shoes for women:
     1. Presto Fly 3 — ₹10,999 — Pink
        Lightweight and breathable running shoes with a flexible sole...

You: What about Adidas?
Bot: Here are some Adidas options:
     1. Superstar Bold 2 — ₹11,999 — Black — Men
        Classic basketball shoes with a modern twist...

You: What was the first product you showed me?
Bot: The first product I showed was the Nike Presto Fly 3...
     [← answered from memory, no Pinecone query fired]
```

---

## ⚡ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | LangGraph (`StateGraph`, `MemorySaver`, `add_messages`) |
| **LLM** | OpenAI `gpt-4o-mini` · `temperature=0` |
| **Embeddings** | OpenAI `text-embedding-3-small` (1536-dim) |
| **Vector Store** | Pinecone Serverless (AWS · `us-east-1` · dotproduct) |
| **SQL Database** | Supabase PostgreSQL (via `psycopg2` · SSL) |
| **Frontend** | Streamlit |
| **Batch Sync** | `tqdm` progress · batches of 100 vectors |
| **Language** | Python 3.12+ |

---

## 🧠 Key Design Decisions

- **History-first routing** eliminates redundant Pinecone queries for follow-up questions. The `NOT_FOUND` sentinel string is a deliberate design choice — deterministic, parser-free, and impossible to confuse with a real product answer.
- **Combined text embedding** (`description + name + brand + gender + price + colour`) gives the vector search rich signal for queries like "pink running shoes under 5000" — far better than embedding just the description.
- **`MemorySaver` keyed by `user_id`** means multiple users can chat simultaneously without their histories bleeding into each other — each thread is isolated.
- **Dual storage (SQL + Vector)** is intentional: Supabase holds the authoritative structured data (filterable, joinable, auditable) while Pinecone handles semantic search. `sync_pinecone.py` is the bridge between them.
- **`temperature=0` for all LLM calls** — both the history check and the product generation use deterministic output. Product recommendations shouldn't be creative.

---

## 🔮 Future Improvements

- [ ] Add price range and brand filter support (structured Pinecone metadata filters)
- [ ] Build a FastAPI backend so the agent can be called from any frontend
- [ ] Add a product recommendation endpoint that takes a user's past queries and suggests items
- [ ] Deploy to Streamlit Cloud or Render
- [ ] Real-time sync — trigger Pinecone upsert automatically when Supabase data changes (via Supabase webhooks)

---

## 👨‍💻 Author

Built for learning: Agentic RAG with LangGraph · Supabase SQL · Pinecone · OpenAI · Streamlit
