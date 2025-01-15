
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import json
from langchain_chroma import Chroma
import constants
import markdown

load_dotenv()

def load_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv('GOOGLE_API_KEY'))

def load_chat_model():
    return ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=os.getenv('GOOGLE_API_KEY'), temperature=0.1)

def generate_metadata_list(document_chunks_count, document_name):
    chunked_data_with_metadata = []
    for idx in range(document_chunks_count):
        metadata = {}
        metadata['title'] = document_name
        metadata['chunk_id'] = f'chunk_{str(idx).zfill(3)}'
        chunked_data_with_metadata.append(metadata)
    return chunked_data_with_metadata
    

def save_vector_embeddings(chroma_db:Chroma, documents, embeddings):
    chroma_db.from_documents(documents,
                             embeddings,
                             collection_name='json_embeddings',
                             persist_directory='vector_embeddings')

def load_vector_embeddings(embeddings):
    vector_store = Chroma(collection_name=constants.EMBEDDINGS_FILE_NAME,
                          embedding_function=embeddings,
                          persist_directory=constants.EMBEDDINGS_PATH)
    return vector_store

def load_prompt(prompt_file_path):
    return prompt_file_path.read().decode("utf-8")

    # with open(prompt_file_path, 'r') as prompt_file:
    #     prompt = prompt_file.read()
    #     return prompt
    
def generate_documents(json_chunks):
    json_chunk_documents = []
    for chunk in json_chunks:
        json_chunk_documents.append({'text':chunk})
    return json_chunk_documents

def fetch_json_file_names(chroma_db:Chroma):
    json_file_names = set()
    content = chroma_db.get(include=['metadatas'])
    metadatas = content['metadatas']
    for metadata in metadatas:
        if not metadata.get('title') in json_file_names:
            json_file_names.add(metadata['title'])
    return json_file_names

def update_nested_data(data, main_key, nested_key, new_value):
    if main_key in data:
        if nested_key in data[main_key]:
            # If the nested key exists, append the new value to the list
            data[main_key][nested_key].append(new_value)
        else:
            # If the nested key does not exist, create a new list with the new value
            data[main_key][nested_key] = [new_value]
    else:
        # If the main key does not exist, create a new entry with the nested key and value
        data[main_key] = {nested_key: [new_value]}
    return data

def retrieve_json_content(chroma_db:Chroma, document_name):
    json_file_content = {}
    result = chroma_db.get(
                            where={"title": document_name},  # Filter condition
                            include=["documents"]
                           )
    for document in result['documents']:
        outer_dict = json.loads(document)
        data = json.loads(outer_dict['text'])
        json_file_content.update(data)
    
    return json_file_content

def verify_json_content(chroma_db:Chroma, document_name):
    result = chroma_db.get(
                            where={"title": document_name},  # Filter condition
                            include=["metadatas"]
                           )
    return len(result['metadatas']) if result['metadatas'] else None

def save_json_rules_engines(json_data):
    with open(os.path.join(constants.ROOT_PATH, 'data.json'), 'w') as out_file:
        out_file.write(json.dumps(json_data, indent=4))
