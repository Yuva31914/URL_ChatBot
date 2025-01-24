import os
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st

# Initialize LLM with environment variable for API key
llm = OpenAI(
    openai_api_key='key',
    temperature=0, max_tokens=100
)

# Streamlit UI elements
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

# List of URLs from sidebar inputs
urls = [st.sidebar.text_input(f"URL {i + 1}") for i in range(3)]
click = st.sidebar.button("Process URLs")

status = st.empty()
dbvector = None
faiss_index_path = "faiss_index"

# Track previous URLs to detect changes
if 'previous_urls' not in st.session_state:
    st.session_state.previous_urls = []

# Load FAISS index if it exists and URLs haven't changed
if os.path.exists(faiss_index_path) and st.session_state.previous_urls == urls:
    dbvector = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(openai_api_key='key'), allow_dangerous_deserialization=True)
    status.text("FAISS index loaded from local storage.")
else:
    status.text("FAISS index not found or URLs have changed. Please process URLs to create it.")

# Process URLs
if click:
    if any(urls):
        loader = UnstructuredURLLoader(urls=urls)
        status.text("Loading data from URLs...")
        data = loader.load()

        textsplit = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        status.text("Splitting text into chunks...")
        docs = textsplit.split_documents(data)

        embeddings = OpenAIEmbeddings(openai_api_key='key')
        dbvector = FAISS.from_documents(docs, embeddings)
        dbvector.save_local(faiss_index_path)
        st.session_state.previous_urls = urls  # Update previous URLs
        status.text("FAISS index created and saved locally.")
    else:
        status.error("Please enter at least one valid URL.")

# Query FAISS index
if dbvector:
    query = st.text_input("Enter your question:")
    if query:
        qa_chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=dbvector.as_retriever())
        response = qa_chain.invoke(query)
        st.write("Answer:", response.get('answer', 'No answer found'))
        st.write("Sources:", response.get('sources', 'No sources found'))
else:
    st.warning("Process URLs to create the FAISS index first.")
