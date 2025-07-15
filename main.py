import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# --- New Import for local embeddings ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from throttling import apply_rate_limit
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
VECTORSTORE_PATH = "faiss_index"
# Use the same model name as in create_vectorstore.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Application Setup ---
app = FastAPI(
    title="Simple LangChain RAG API",
    description="An API for a conversation with a knowledge base.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class RAGRequest(BaseModel):
    question: str

# --- Load Models and Vector Store ---

# 1. Verify API Key (still needed for the Gemini Chat Model)
if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Gemini API key not found in environment variables.")
api_key = os.getenv("GEMINI_API_KEY")

# 2. Initialize Chat Model (this part is unchanged)
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    google_api_key=api_key
)

# 3. Initialize the same local embedding model used for indexing
print(f"Initializing local embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
)

# 4. Load the Pre-built FAISS Vector Store from Disk
if not os.path.exists(VECTORSTORE_PATH):
    raise FileNotFoundError(
        f"Vector store not found at {VECTORSTORE_PATH}. "
        "Please run 'python create_vectorstore.py' to create it first."
    )

print(f"Loading vector store from {VECTORSTORE_PATH}...")
vector_db = FAISS.load_local(
    VECTORSTORE_PATH, 
    embeddings=embedding_model, 
    allow_dangerous_deserialization=True
)
print("Vector store loaded successfully.")

# 5. Create the retriever (this part is unchanged)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})


# --- LangChain RAG Chain Definition (Unchanged) ---
# ... (The rest of your main.py file is exactly the same) ...

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful medical assistant that answers questions on matters health 
         based on the provided context. Use a helpful and concise tone. If the context does not provide 
         enough information, feel free to say that you don't know the answer.
         If the context is not relevant to the question, you may decline to answer stating that the context does not
         provide the necessary information. Never answer questions that may be harmful or illegal.
         The context is provided below. If the context is empty, you may say that you don't have any information to 
         answer the question.
         Context: {context}"""
         ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chat_history = ChatMessageHistory()

def get_history(input_dict):
    return chat_history.messages

rag_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("input") | retriever,
        history=get_history,
    )
    | prompt
    | chat_model
    | StrOutputParser()
)

# --- API Endpoints (Unchanged) ---

@app.get("/")
async def root():
    return {"status": "ok", "message": "Simple RAG API is running."}

@app.post("/clear_memory")
async def clear_memory():
    """Clears the conversation history."""
    global chat_history
    chat_history.clear()
    return {"status": "ok", "message": "Conversation memory has been cleared."}

@app.post("/ask_rag")
async def ask_rag_endpoint(request: RAGRequest):
    user_input = request.question
    apply_rate_limit("global_unauthenticated_user")
    response = rag_chain.invoke({"input": user_input})
    chat_history.add_message(HumanMessage(content=user_input))
    chat_history.add_message(AIMessage(content=response))
    print("Current History:", chat_history.messages)
    return {"answer": response}