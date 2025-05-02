# News-Research-Tool

This is a web-based tool built with **Streamlit** and **LangChain** that allows users to input multiple news article URLs and ask questions about their content. The app uses **OpenAI embeddings** and **FAISS** for vector-based retrieval, enabling source-cited natural language answers via a **Retrieval-Augmented Generation (RAG)** pipeline.

## 🚀 Features
- Input up to 3 news article URLs
- Automatic loading, chunking, and embedding of content
- Local vector store using FAISS for efficient retrieval
- Natural language question answering with sources
- Simple and interactive UI via Streamlit

## 🛠️ Tech Stack
- Python
- Streamlit
- LangChain
- OpenAI API
- FAISS
- dotenv

## 📦 Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/news-research-tool.git
   cd news-research-tool

2.  ##Install dependencies
    Install the required Python packages by running the following command:
    ```bash
    pip install -r requirements.txt

3.  Set up environment variables
    Create a .env file in the root directory and add your OpenAI API key:
    ```bash
    OPENAI_API_KEY=your_openai_api_key_here

4.  Run the Streamlit app
    Start the application with:
    ```bash
    streamlit run main.py


