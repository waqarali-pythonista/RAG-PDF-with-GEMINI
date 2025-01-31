import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------- Page Configurations ----------------------
st.set_page_config(
    page_title="🔍 RAG-powered Q&A",
    page_icon="🤖",
    layout="wide",
)

# ---------------------- Custom CSS for Styling ----------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        font-family: Arial, sans-serif;
    }
    .stApp {
        background-color: black;
        border-radius: 10px;
        padding: 20px;
    }
    .stTitle {
        font-size: 30px;
        font-weight: bold;
        color: #4CAF50;
    }
    .chat-container {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 10px;
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    .bot-message {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    .loading {
        font-size: 18px;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Title & Sidebar ----------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6f/Emoji_u1f916.svg", width=70)
st.sidebar.title("🤖 RAG Chatbot")
st.sidebar.write("🔍 **Ask anything about YOLOv9!**")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Built with LangChain + Google Gemini AI")

st.title("📄 RAG-powered Q&A on YOLOv9")

# ---------------------- Load & Process PDF ----------------------
st.sidebar.subheader("📂 Loading PDF...")
loader = PyPDFLoader("yolov9_paper.pdf")
data = loader.load()
st.sidebar.success("✅ PDF Loaded Successfully!")

st.sidebar.subheader("📄 Splitting Document...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
st.sidebar.success(f"✅ Split into {len(docs)} chunks!")

# ---------------------- Vector Embeddings ----------------------
st.sidebar.subheader("📡 Creating Embeddings...")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
st.sidebar.success("✅ Embeddings Created!")

# ---------------------- Chatbot LLM Setup ----------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

# ---------------------- Prompt Template ----------------------
system_prompt = (
    "You are an AI assistant specialized in answering questions about YOLOv9. "
    "Use the provided context to answer the query. "
    "If the answer is unknown, say 'I don't know.' Keep answers concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ---------------------- Chat Input & Processing ----------------------
query = st.chat_input("💬 Ask something about YOLOv9...") 

if query:
    # Display user's message
    st.markdown(f'<div class="user-message">👤 **You:** {query}</div>', unsafe_allow_html=True)
    
    # Loading Animation
    with st.spinner("🔍 Thinking..."):
        time.sleep(2)  # Simulate processing time
    
        # RAG Chain Processing
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]

    # Display AI's response
    st.markdown(f'<div class="bot-message">🤖 **Gemini AI:** {answer}</div>', unsafe_allow_html=True)
