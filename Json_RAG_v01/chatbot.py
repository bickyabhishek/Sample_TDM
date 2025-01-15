import os
import streamlit as st
import constants
import utils
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import re

llm = utils.load_chat_model()
embeddings = utils.load_embeddings()

chroma_db = utils.load_vector_embeddings(embeddings)

def app():
    st.title("Synthetic Data Generator")
    st.markdown(
        """
        ### Features:
        1. Define conditions for synthetic data generation.
        2. Upload a JSON file with rules and engines.
        3. Leverage Google Generative AI via LangChain for data synthesis.
        """
    )

    json_file_names_list = utils.fetch_json_file_names(chroma_db)
    json_file_name = st.selectbox('JSON File Name Selection', json_file_names_list)
    json_data = utils.retrieve_json_content(chroma_db, json_file_name)
    utils.save_json_rules_engines(json_data)
    with open(os.path.join(constants.ROOT_PATH, 'data.json'), 'r') as rules_engine_file:
        rules_and_engines = json.load(rules_engine_file)
    json_rules = json.dumps(rules_and_engines, indent=4) if rules_and_engines else "No rules provided."

    prompt_file_uploader = st.file_uploader('Upload Prompt File', type=['txt'])
    if prompt_file_uploader:
        prompt_template = utils.load_prompt(prompt_file_uploader)
        excel_file_name = json_file_name.split('.')[0]+'.xlsx'
        # Format the prompt
        prompt = PromptTemplate(
            input_variables=["json_rules", "excel_file_name"],
            template=prompt_template
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        if st.button("Generate Synthetic Data"):
            try:
                # Call the chain with user input
                result = llm_chain.run(json_rules=json_rules, excel_file_name=excel_file_name)
                code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", result, re.DOTALL)
                with open('data_extract.py', 'w') as code_file:
                    for code in code_blocks:
                        code_file.write(code)
                success_code = os.system('py data_extract.py')
                if os.path.exists(excel_file_name):
                    excel_data = pd.ExcelFile(excel_file_name)
                    sheet_names = excel_data.sheet_names  # Get all sheet names
                    # Display sheet options
                    st.write("Available Sheets:", sheet_names)

                    st.markdown('---')

                    with st.container(border=True):
                        col1, _, col2 = st.columns([0.45,0.1,0.45])

                        with col1:
                            df1 = pd.read_excel(excel_file_name, sheet_name='Header',  engine='openpyxl')
                            st.write(f"Displaying Sheet: {'Header'}")
                            st.dataframe(df1)

                        with col2:
                            df2 = pd.read_excel(excel_file_name, sheet_name='Line',  engine='openpyxl')
                            st.write(f"Displaying Sheet: {'Line'}")
                            st.dataframe(df2)
                else:
                    st.error('Unable to Generate the Synthetic Data..')
                
            except Exception as e:
                st.error(f"Error generating synthetic data: {e}")
