import os
import streamlit as st
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import  ChatGoogleGenerativeAI
from dotenv import load_dotenv
import re
import pandas as pd

load_dotenv()

# Initialize the LangChain model
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key=os.getenv('GOOGLE_API_KEY'))


# Streamlit UI
st.set_page_config(page_title='Synthetic Data Generator', layout='wide')
st.title("Synthetic Data Generator")
st.markdown(
    """
    ### Features:
    1. Define conditions for synthetic data generation.
    2. Upload a JSON file with rules and engines.
    3. Leverage Google Generative AI via LangChain for data synthesis.
    """
)


# # Input for user-defined conditions
# st.subheader("Define Conditions for Synthetic Data Generation")
# user_conditions = st.text_area("Enter the conditions for generating synthetic data:")

# File uploader for JSON rules and engines
st.subheader("Upload JSON File with Rules and Engines")
uploaded_file = st.file_uploader("Choose a JSON file", type="json")

user_conditions = "Ensure the data adheres to the specified rules and engine configurations and generated code should contain logic to parse the json file named data.json for all the columns in json file and also logic for writing header data and line data into single **xlsx** file called sample.xlsx. Generate data for all the columns in Json file."


# Parse the JSON file if uploaded
if uploaded_file is not None:
    try:
        rules_and_engines = json.load(uploaded_file)
        st.success("JSON file successfully loaded!")
        #st.json(rules_and_engines)
    except Exception as e:
        st.error(f"Error parsing JSON file: {e}")
        rules_and_engines = None
else:
    rules_and_engines = None

prompt_upload = st.file_uploader("Upload a prompt file", type="txt")
if prompt_upload is not None:
    try:
        prompt_text = prompt_upload.getvalue().decode("utf-8")
        st.success("Prompt file successfully loaded!")
    except Exception as e:
        st.error(f"Error parsing prompt file: {e}")
        prompt_text = None
else:
    prompt_text = None

if prompt_text is not None:
    prompt_temp = prompt_text + "\n\n{json_rules}" + "\n\n"+ user_conditions
    prompt_template = f"""{prompt_temp}"""
else:
    prompt_template = f""" """


# """

json_rules = json.dumps(rules_and_engines, indent=4) if rules_and_engines else "No rules provided."

# Format the prompt
prompt = PromptTemplate(
    input_variables=["json_rules"],
    template=prompt_template
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

if st.button("Generate Synthetic Data"):
    try:
        # Call the chain with user input
        result = llm_chain.run(json_rules=json_rules)
        code_blocks = re.findall(r"```(?:\w+\n)?(.*?)```", result, re.DOTALL)
        with open('data_extract.py', 'w') as code_file:
            for code in code_blocks:
                code_file.write(code)
        success_code = os.system('py data_extract.py')
        if os.path.exists('sample.xlsx'):
            excel_data = pd.ExcelFile('sample.xlsx')
            sheet_names = excel_data.sheet_names  # Get all sheet names
            # Display sheet options
            st.write("Available Sheets:", sheet_names)

            st.markdown('---')

            with st.container(border=True):
                col1, _, col2 = st.columns([0.45,0.1,0.45])

                with col1:
                    df1 = pd.read_excel('sample.xlsx', sheet_name=sheet_names[0],  engine='openpyxl')
                    st.write(f"Displaying Sheet: {'Header'}")
                    st.dataframe(df1)

                with col2:
                    df2 = pd.read_excel('sample.xlsx', sheet_name=sheet_names[1],  engine='openpyxl')
                    st.write(f"Displaying Sheet: {'Line'}")
                    st.dataframe(df2)
        else:
            st.error('Unable to Generate the Synthetic Data..')
        
    except Exception as e:
        st.error(f"Error generating synthetic data: {e}")
