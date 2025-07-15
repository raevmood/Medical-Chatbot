### Medical RAG Chatbot
A private, locally-run chatbot that answers questions based on a provided medical document. It uses a Retrieval-Augmented Generation (RAG) pipeline to ensure answers are accurate and context-aware.

## Goal
The objective of this project is to provide a secure and private way to interact with a knowledge base (a medical book). User data and the source document are kept private by generating embeddings locally, while leveraging a powerful LLM (Google's Gemini) for high-quality conversational responses.

## Features
1. Private & Secure: Document content is processed locally and is never sent to an external API.
2. Accurate Responses: Implements a RAG pipeline to base answers on the provided text, reducing hallucinations.
3. Web Interface: Simple and intuitive chat interface built with Flask.
4. Hybrid Model Approach: Uses local Sentence Transformers for embeddings and the Gemini API for generation.

## Getting Started
Follow these instructions to get a copy of the project up and running on your local machine.

# Prerequisites
Python 3.8+
Git
Installation & Setup
Clone the repository:
Generated sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Create and activate a virtual environment:
# For Windows
python -m venv venv
.\venv\Scripts\activate
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Install the required dependencies:
Generated sh
pip install -r requirements.txt
(Note: You will need to create a requirements.txt file by running pip freeze > requirements.txt)
Set up your environment variables:
Create a file named .env in the project root.
Add your Google Gemini API key to it:
Generated env
GEMINI_API_KEY="YOUR_API_KEY_HERE"
Add your knowledge base:
Place your source document (e.g., a medical textbook) in the root directory and name it Medical_book.pdf.
Running the Application
Create the Vector Store (One-time setup):
Run the indexing script. This reads your PDF, creates embeddings, and saves them locally.
python create_vectorstore.py
A folder named faiss_index will be created.
Run the Flask Web Server:
python main.py
# Access the Chatbot:
Open your web browser and navigate to: http://127.0.0.1:8080