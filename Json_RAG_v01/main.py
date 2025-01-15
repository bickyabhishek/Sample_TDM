import streamlit as st
from streamlit_option_menu import option_menu
import embeddings, chatbot

st.set_page_config('Chatbot', layout='wide')

def get_options():
    options = ['Upload Documents', 'Chatbot']
    icons = ['database', 'robot']
    return options, icons

with st.sidebar:
    options, icons = get_options()
    option_selected = option_menu(
        menu_title=None,
        options=options,
        menu_icon=None,
        icons = icons,
        styles={
                    "container": {"padding": "0!important", "background-color": "#34495E"},
                    "icon": {"color": "white", "font-size": "20px"},
                    "nav-link": {
                        "font-size": "20px",
                        "font-family": "system-ui",
                        "text-align": "left",
                        "margin": "5px",
                        "font-weight": "bold",
                        "--hover-color": "#E74C3C"
                    },
                    "nav-link-selected": {"background-color": "#28B463"}
                }
    )

with st.container():
    col1,col2 = st.columns([0.05, 0.95])
    with col1:
        st.image("./images/chatbot.png", width=100)
    with col2:
        st.markdown("<h1 style='text-align: center; color: orange;'>RAG Based Chatbot</h1>", unsafe_allow_html=True)
    
    #st.markdown("<h1 style='text-align: center; color: orange;'>RAG Based Chatbot</h1>", unsafe_allow_html=True)
st.markdown('---')

if option_selected == 'Upload Documents':
    embeddings.app()
elif option_selected == 'Chatbot':
    chatbot.app()
