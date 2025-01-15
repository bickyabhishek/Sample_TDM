import os

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
EMBEDDINGS_PATH = os.path.join(ROOT_PATH, 'vector_embeddings')
EMBEDDINGS_FILE_NAME = 'json_embeddings'
PROMPT_FILE_NAME = 'prompt_template.txt'
DATABASE_NAME = 'chroma.sqlite3'
