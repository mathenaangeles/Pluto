import os
import nltk
import streamlit as st
from llama_index.core.tools import FunctionTool,  QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

st.set_page_config(page_title='Pluto',page_icon = 'images/pluto_icon.png', initial_sidebar_state = 'auto')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk_data_dir = "./nltk_cache/"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.clear()
nltk.data.path.append(nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

def save_uploaded_file(uploaded_file):
    with open(os.path.join('./data', uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

llm = Ollama(model="llama3", request_timeout=3600)
embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

Settings.llm = llm 
Settings.embed_model = embed_model

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing documents..."):
        reader = SimpleDirectoryReader("./data", recursive=True)
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=Settings)
        return index
index = load_data()
chat_engine = index.as_chat_engine(chat_mode="context", llm=llm)
query_engine = index.as_query_engine()
query_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="pluto_rag",
    description="This is a RAG engine that generates responses based on the PDF data, if it exists in the database.",
)

def compute_valuation(discounted_cash_flow: int, patent_value: int, risk: int) -> int:
    return discounted_cash_flow + patent_value - risk
compute_valuation_tool = FunctionTool.from_defaults(fn=compute_valuation)


agent = ReActAgent.from_tools(
    [query_tool, compute_valuation_tool], llm=llm, verbose=True
)

uploaded_files = st.file_uploader(
    label="Upload files to the data directory.",
    accept_multiple_files=True,
    key="file_uploader",
    help="You can upload multiple files.",
    on_change=None,
    disabled=False,
    label_visibility="visible"
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_uploaded_file(uploaded_file)
        st.success(f"File {uploaded_file.name} uploaded successfully!")

def main():
    st.logo('images/pluto.png', link="https://github.com/mathenaangeles/Pluto", icon_image='images/pluto_icon.png')
    st.title(":robot_face: Valuation Agent")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I will assist you in creating a business valuation report. Which company would you like to generate a report for?"}
        ]
    if prompt := st.chat_input("Write your message here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.chat(st.session_state.messages[-1]["content"])
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
