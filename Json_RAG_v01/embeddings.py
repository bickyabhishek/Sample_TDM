import os
import streamlit as st
import json
import utils
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_chroma import Chroma
import constants


embeddings = utils.load_embeddings()

chroma_db = Chroma(collection_name = constants.EMBEDDINGS_FILE_NAME,
                   embedding_function= embeddings,
                   persist_directory=constants.EMBEDDINGS_PATH,
                   create_collection_if_not_exists=True,
                   collection_metadata=None)

def app():
    json_file_loaded = st.file_uploader('Upload JSON Documents',type=['json'])
    if json_file_loaded:
        data_presence_check = utils.verify_json_content(chroma_db, json_file_loaded.name)
        if data_presence_check is None:
            data = json.load(json_file_loaded)
            splitter = RecursiveJsonSplitter(max_chunk_size=30000)
            json_chunks = splitter.split_text(data)
            json_chunk_documents = utils.generate_documents(json_chunks)
            metadata_list = utils.generate_metadata_list(len(json_chunks), json_file_loaded.name)
            documents = splitter.create_documents(json_chunk_documents, metadatas=metadata_list)
            for document in documents:
                print(document.page_content)
            if st.button('Calculate Embeddings'):
                with st.spinner('Embeddings Calculation In Progress..'):
                    utils.save_vector_embeddings(chroma_db, documents, embeddings)
                    if os.path.exists(os.path.join(constants.EMBEDDINGS_PATH,constants.DATABASE_NAME)):
                        st.success('Embeddings Calculated Successfully..')
        else:
            st.error('Document is already present in the Database')

