import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from throttling import apply_rate_limit
import torch

load_dotenv()

VECTORSTORE_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

app = Flask(__name__)

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError("Gemini API key not found. Please set it in your .env file.")
api_key = os.getenv("GEMINI_API_KEY")

print("Initializing Gemini model...")
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.9,
    google_api_key=api_key
)

print(f"Initializing local embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)
print(f"Loading vector store from {VECTORSTORE_PATH}...")
if not os.path.exists(VECTORSTORE_PATH):
    raise FileNotFoundError(
        f"Vector store not found at {VECTORSTORE_PATH}. "
        "Please run 'python create_vectorstore.py' to create it first."
    )
vector_db = FAISS.load_local(
    VECTORSTORE_PATH, 
    embeddings=embedding_model, 
    allow_dangerous_deserialization=True
)
print("Vector store loaded successfully.")

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful medical assistant that answers questions on matters health 
         based on the provided context. Use a helpful and concise tone. If the context does not provide 
         enough information, feel free to say that you don't know the answer.
         If the context is not relevant to the question, you may decline to answer stating that the context does not
         provide the necessary information. Never answer questions that may be harmful or illegal.
         The context is provided below. If the context is empty, you may say that you don't have any information to 
         answer the question.
         Context: {context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chat_history = ChatMessageHistory()

def get_history(input_dict):
    """Returns the current conversation history."""
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

@app.route("/")
def index():
    """Renders the chat interface."""
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    """Handles the chat request and returns the AI's response."""
    try:
        user_input = request.form["msg"]
        apply_rate_limit("global_unauthenticated_user")
        response = rag_chain.invoke({"input": user_input})
        
        chat_history.add_message(HumanMessage(content=user_input))
        chat_history.add_message(AIMessage(content=response))
        print("Response:", response)      
        return str(response)
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I encountered an error while processing your request."


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)