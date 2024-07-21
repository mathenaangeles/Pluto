import streamlit as st
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

llm = Ollama(model="llama3", request_timeout=3600)
embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

Settings.llm = llm 
Settings.embed_model = embed_model

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Please wait while I load and index the documents..."):
        reader = SimpleDirectoryReader("./data", recursive=True)
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=Settings)
        return index
index = load_data()
chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

# def valuation(patent_value: int, risk: int) -> int:
#     return patent_value - risk
# valuation_tool = FunctionTool.from_defaults(fn=valuation)
# valuation_agent = ReActAgent.from_tools(
#     [valuation_tool],
#     llm=llm,
#     verbose=True,
# )
# def get_valuation_response():
#     response = valuation_agent.chat("What is the value given a patent value of 10 and a risk of 5?")
#     return response

def main():
    st.title("Valuation Agent")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am a valuation agent that will assist you in creating a business valuation report. Which company would you like to generate a report for?"}
        ]
    # response = get_valuation_response()
    # st.markdown(f"<p>{response}</p>", unsafe_allow_html=True)
    if prompt := st.chat_input("Write your message here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Please wait while I think."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
