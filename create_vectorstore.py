import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# --- New Import for local embeddings ---
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables (still needed for the main app's chat model)
load_dotenv()

# --- Configuration ---
PDF_PATH = "Medical_book.pdf"
VECTORSTORE_PATH = "faiss_index"

# --- Define the local embedding model ---
# We will use a popular, well-performing model from Hugging Face.
# "all-MiniLM-L6-v2" is a great balance of speed and quality.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_and_save_vectorstore():
    """
    This function performs the one-time process of:
    1. Loading the PDF document.
    2. Splitting it into chunks.
    3. Generating embeddings for each chunk using a local model.
    4. Creating a FAISS vector store.
    5. Saving the vector store to disk.
    """
    print("Starting the indexing process...")

    # 1. Check if the PDF file exists
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"The specified PDF file was not found: {PDF_PATH}")
    
    print(f"Loading document: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks of text.")

    # --- Use the local Sentence Transformer model ---
    print(f"Initializing local embedding model: {EMBEDDING_MODEL_NAME}")
    # The first time you run this, it will download the model from Hugging Face.
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
    )

    print("Creating FAISS vector store from chunks... (This may take a moment)")
    # Create the FAISS index from the document chunks and local embeddings
    vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)

    print(f"Saving vector store to: {VECTORSTORE_PATH}")
    # Save the created index to a local folder
    vector_db.save_local(VECTORSTORE_PATH)

    print("--- Indexing complete! ---")

if __name__ == "__main__":
    create_and_save_vectorstore()