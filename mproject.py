import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
import time

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit app title
st.title("Smart AgroCare Chat Bot")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
"""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
<context>
{context}
<context>
Questions: {input}
"""
)

# Define a function to create vector embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("embed_data")  # Data ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

# Sidebar content
with st.sidebar:
    st.title("AgroCare")
    st.subheader("This app clears your doubts about potato crops[👉]")
    add_vertical_space(2)
    st.write("")

# Main app logic
# if st.button("Tap to make bot Ready"):
vector_embedding()
    # st.write("Go........! 🚀") #This is means embeddings are ready

# Chat functionality
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferWindowMemory(k=2)
    )

if prompt1 := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt1})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Document question answering
if prompt1:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            print("Response time:", time.process_time() - start)
            # st.write(response)  # Print the entire response for debugging
            if isinstance(response, dict) and 'answer' in response:
                st.write(response['answer'])
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            else:
                st.write("Unexpected response format")
                st.session_state.messages.append({"role": "assistant", "content": "Unexpected response format"})