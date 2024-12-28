import streamlit as st
import os
import shutil
from dotenv import load_dotenv, set_key
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma as ChromaLangChain
import chromadb
import pandas as pd
 
# Specify the relative path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
persist_directory = "./chroma_db"
load_dotenv(dotenv_path)
client = chromadb.PersistentClient(path=persist_directory)
 
 
# Helper function to update .env file
def update_dotenv(key, value):
    set_key(dotenv_path, key, value)
    os.environ[key] = value
 
 
# Initialize Azure OpenAI embeddings
def initiate_azure_embeddings():
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_vector_embedding_deployment_name = os.getenv("AZURE_VECTOR_EMBEDDING_DEPLOYMENT_NAME")
    openai_api_version = os.getenv("OPENAI_API_VERSION")
 
    if not all([azure_openai_api_key, azure_endpoint, azure_vector_embedding_deployment_name, openai_api_version]):
        raise ValueError("Missing Azure OpenAI configurations in the .env file.")
 
    return AzureOpenAIEmbeddings(
        openai_api_version=openai_api_version,
        openai_api_type="azure",
        openai_api_key=azure_openai_api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_vector_embedding_deployment_name
    )
 
 
# Initialize Azure OpenAI chat model
def initiate_azure_chat_openai():
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    azure_chatai_deployment_name = os.getenv("AZURE_CHATAI_DEPLOYMENT_NAME")
    openai_api_version = os.getenv("OPENAI_API_VERSION")
 
    return AzureChatOpenAI(
        openai_api_version=openai_api_version,
        openai_api_type="azure",
        openai_api_key=azure_openai_api_key,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_chatai_deployment_name
    )
 
 
# Function to create and query Tree Index
def create_tree_index(documents):
    embeddings = initiate_azure_embeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    print(embeddings)
 
    # Process and split documents
    split_docs = text_splitter.create_documents(documents)
    vectordb = ChromaLangChain.from_documents(split_docs, embeddings, persist_directory=persist_directory)
 
    
    # Tree Index Query Engine (using RetrievalQA)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=initiate_azure_chat_openai(),
        retriever=retriever,
        chain_type="stuff"
    )
    print(chain)
    return chain
 
 
# Clean vector database
def clean_vectordb():
    try:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        st.success("Vector database cleaned successfully.")
    except Exception as e:
        st.error(f"Failed to clean vector database: {str(e)}")
 
 
# UI Header Page
def ui_header_page():
    st.title("ESAN User")
    # st.markdown(
    #     "This application leverages **Azure OpenAI** and **LangChain** with a **Tree Index Query Engine** "
    #     "to efficiently process and query large text data."
    # )
 
 
# Upload and Process Files
def process_uploaded_files(json_file, xlsx_file):
    documents = []
 
    # Process JSON File
    if json_file is not None:
        try:
            json_text = [json_file.read().decode()]
            documents.extend(json_text)
        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")
 
    # Process Excel File
    if xlsx_file is not None:
        try:
            xlsx_data = pd.read_excel(xlsx_file)
            xlsx_text = xlsx_data.apply(lambda row: " ".join(row.values.astype(str)), axis=1).tolist()
            documents.extend(xlsx_text)
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
 
    return documents
 
 
# Streamlit Sidebar
def ui_sidebar():
    with st.sidebar:
        st.subheader("Options")
        if st.button("Clean Vector Database"):
            clean_vectordb()
            st.experimental_rerun()
 
 
# Main Function
def main():
    ui_header_page()
    ui_sidebar()
 
    st.write("## Upload Files to Generate Tree Index")
    json_file = st.file_uploader("Upload a JSON file", type=["json"])
    xlsx_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
 
    if st.button("Generate Tree Index"):
        documents = process_uploaded_files(json_file, xlsx_file)
        if documents:
            st.session_state.chain = create_tree_index(documents)
            st.success("Tree Index successfully generated!")
        else:
            st.error("No valid documents uploaded.")
 
    # User Query Interface
    if "chain" in st.session_state:
        st.write("## Query the Tree Index")
        user_query = st.text_area("Enter your query:")
        if st.button("Generate Response"):
            try:
                response = st.session_state.chain.run(user_query)
                st.text_area("Response:", response, height=200)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
 
 
if __name__ == "__main__":
    main()
