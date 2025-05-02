import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (like OpenAI API key)
load_dotenv()

# UI Title and Sidebar
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Process button
process_url_clicked = st.sidebar.button("Process URLs")
vectorstore_path = "faiss_index_store"
main_placeholder = st.empty()

# LLM Configuration
llm = OpenAI(temperature=0.9, max_tokens=500)

# Process the URLs and build the vectorstore
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embeddings...✅✅✅")
    time.sleep(2)

    # Save vectorstore
    vectorstore_openai.save_local(vectorstore_path)
    main_placeholder.text("Saved Vectorstore Successfully ✅")

# Question input and query
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(vectorstore_path):
        # Load vectorstore safely
        vectorstore = FAISS.load_local(
            vectorstore_path,
            OpenAIEmbeddings(),
            allow_dangerous_deserialization=True  # ✅ Required for pickle safety
        )

        # Run the retrieval chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
