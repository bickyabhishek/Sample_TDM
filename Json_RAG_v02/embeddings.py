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

def process_json_data(json_file_loaded, chroma_db:Chroma, process_flag):
    data = json.load(json_file_loaded)
    splitter = RecursiveJsonSplitter(max_chunk_size=30000)
    json_chunks = splitter.split_text(data)
    if process_flag == 'Modify':
        with st.spinner('Processing..'):
            json_chunk_documents = utils.generate_documents(json_chunks)
            metadata_list = utils.generate_metadata_list(len(json_chunks), json_file_loaded.name)
            documents = splitter.create_documents(json_chunk_documents, metadatas=metadata_list)
            utils.save_vector_embeddings(chroma_db, documents, embeddings)
            chunk_ids_list = utils.get_ids_list(chroma_db, json_file_loaded.name)
            if chunk_ids_list is not None:
                st.success(f'Embeddings have been Successfully Updated for JSON File - {json_file_loaded.name}')
            else:
                st.error(f'Error in Embeddings Updation Process')
    elif process_flag == 'Upload':
        with st.spinner('Processing..'):
            json_chunk_documents = utils.generate_documents(json_chunks)
            metadata_list = utils.generate_metadata_list(len(json_chunks), json_file_loaded.name)
            documents = splitter.create_documents(json_chunk_documents, metadatas=metadata_list)
            utils.save_vector_embeddings(chroma_db, documents, embeddings)
            chunk_ids_list = utils.get_ids_list(chroma_db, json_file_loaded.name)
            if chunk_ids_list is not None:
                st.success(f'Embeddings for JSON File - {json_file_loaded.name} generated successfully..')
            else:
                st.error('Error in Embeddings Calculation Process..')

def app():
    json_file_loaded = st.file_uploader('Upload JSON Documents',type=['json'])
    if json_file_loaded:
        chunk_ids_list = utils.get_ids_list(chroma_db, json_file_loaded.name)
        if chunk_ids_list is not None:
            col1, col2 = st.columns([0.7, 0.3])
            with col1.container():
                modify_col, delete_col = st.columns(2)
                with modify_col:
                    if st.button('Modify'):
                        chunk_ids_list = utils.get_ids_list(chroma_db, json_file_loaded.name)
                        if chunk_ids_list is not None:
                            utils.delete_json_content(chroma_db, chunk_ids_list)
                            process_json_data(json_file_loaded, chroma_db, 'Modify')
                        else:
                            st.error(f'Embeddings of JSON File - {json_file_loaded.name} is not present in the Database.')
                with delete_col:
                    if st.button('Delete'):
                        chunk_ids_list = utils.get_ids_list(chroma_db, json_file_loaded.name)
                        utils.delete_json_content(chroma_db, chunk_ids_list)
                        chunk_ids_list = utils.get_ids_list(chroma_db, json_file_loaded.name)
                        if chunk_ids_list is None:
                            st.error(f'Successfully deleted the JSON File - {json_file_loaded.name}')
        else:
            if st.button('Upload Embeddings'):
                process_json_data(json_file_loaded, chroma_db, 'Upload')