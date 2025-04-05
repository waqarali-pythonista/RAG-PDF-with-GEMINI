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
from langchain_core.documents import Document
import uuid
import chromadb
import os
from PIL import Image
import io
import base64
import fitz  # PyMuPDF
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
from google.api_core import exceptions
from difflib import SequenceMatcher

# Define highlight_text function
def highlight_text(text, query):
    """Highlight parts of the text that are most similar to the query"""
    # Split query into words and filter out common words
    query_words = set(query.lower().split())
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    query_words = query_words - stop_words
    
    # Split text into words and create spans
    words = text.split()
    highlighted_words = []
    
    for word in words:
        # Check if word is similar to any query word
        should_highlight = False
        clean_word = word.lower().strip('.,!?()[]{}":;')
        
        for query_word in query_words:
            if (query_word in clean_word or 
                clean_word in query_word or 
                SequenceMatcher(None, clean_word, query_word).ratio() > 0.8):
                should_highlight = True
                break
        
        if should_highlight:
            highlighted_words.append(f"<mark>{word}</mark>")
        else:
            highlighted_words.append(word)
    
    return " ".join(highlighted_words)

# Load environment variables
load_dotenv()

# ---------------------- Initialize Google API ----------------------
# Configure Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your GOOGLE_API_KEY in the .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.3,
    convert_system_message_to_human=True,
    google_api_key=GOOGLE_API_KEY,
)

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
        color: black;
    }
    .bot-message {
        background-color: #F1F0F0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
        color: black;
    }
    .loading {
        font-size: 18px;
        color: #777;
    }
    mark {
        background-color: #fff3cd;
        padding: 0.2em 0.4em;
        border-radius: 4px;
        font-weight: 500;
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
pdf_path = "yolov9_paper.pdf"

# Load PDF with PyPDFLoader
loader = PyPDFLoader(pdf_path)
pages = loader.load()
st.sidebar.success("✅ PDF Loaded Successfully!")

st.sidebar.subheader("📄 Processing Document...")

# Create text splitter for better content handling
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

# Process each page with better metadata handling
processed_docs = []
page_docs = []  # Store complete pages
pdf_document = fitz.open(pdf_path)

for i, page in enumerate(pages):
    page_num = i + 1
    
    # Extract text and metadata from the PDF page
    pdf_page = pdf_document[i]
    text = pdf_page.get_text()
    
    # Extract section information
    lines = text.split('\n')
    section_info = ""
    for line in lines:
        if any(line.strip().startswith(prefix) for prefix in ['1.', '2.', '3.', '4.']):
            section_info = line.strip()
            break
    
    # Create a document for the entire page with unique ID
    page_doc = Document(
        page_content=text,
        metadata={
            "id": f"page_{page_num}_{uuid.uuid4().hex}",
            "page_number": page_num,
            "type": "full_page",
            "section": section_info,
            "source": pdf_path,
            "total_pages": len(pages)
        }
    )
    page_docs.append(page_doc)
    
    # Split content into chunks for more granular retrieval
    chunks = text_splitter.split_text(text)
    
    # Create documents with enhanced metadata
    for chunk_idx, chunk in enumerate(chunks):
        processed_doc = Document(
            page_content=chunk,
            metadata={
                "id": f"chunk_{page_num}_{chunk_idx}_{uuid.uuid4().hex}",
                "page_number": page_num,
                "chunk_index": chunk_idx,
                "type": "chunk",
                "section": section_info,
                "source": pdf_path,
                "total_pages": len(pages)
            }
        )
        processed_docs.append(processed_doc)

pdf_document.close()
st.sidebar.success(f"✅ Processed {len(processed_docs)} text chunks and {len(page_docs)} pages!")

# ---------------------- Vector Store Setup ----------------------
st.sidebar.subheader("📡 Creating Embeddings...")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def setup_vectorstore(chunk_docs, page_docs):
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Use a simpler database path
        db_path = "chroma_db"
        
        # Try to remove the old database if it exists
        if os.path.exists(db_path):
            try:
                # Force close any open SQLite connections
                import sqlite3
                from sqlite3 import OperationalError
                try:
                    conn = sqlite3.connect(f"{db_path}/chroma.sqlite3")
                    conn.close()
                except (OperationalError, sqlite3.Error):
                    pass
                
                # Remove directory with multiple attempts
                for _ in range(3):
                    try:
                        import shutil
                        shutil.rmtree(db_path, ignore_errors=True)
                        time.sleep(2)  # Wait longer between attempts
                        if not os.path.exists(db_path):
                            break
                    except Exception:
                        time.sleep(2)
                        continue
            except Exception as e:
                st.warning(f"Database cleanup warning (non-critical): {str(e)}")
        
        # Ensure the directory exists
        os.makedirs(db_path, exist_ok=True)
        
        # Create vectorstore for chunks with IDs
        chunk_vectorstore = Chroma.from_documents(
            documents=chunk_docs,
            embedding=embeddings,
            persist_directory=os.path.join(db_path, "chunks"),
            collection_name="chunks",
            ids=[doc.metadata["id"] for doc in chunk_docs]
        )
        
        # Create vectorstore for pages with IDs
        page_vectorstore = Chroma.from_documents(
            documents=page_docs,
            embedding=embeddings,
            persist_directory=os.path.join(db_path, "pages"),
            collection_name="pages",
            ids=[doc.metadata["id"] for doc in page_docs]
        )
        
        return chunk_vectorstore, page_vectorstore
        
    except Exception as e:
        st.error(f"Detailed error in vector store setup: {str(e)}")
        raise

try:
    chunk_vectorstore, page_vectorstore = setup_vectorstore(processed_docs, page_docs)
    
    # Configure retrievers with simpler parameters
    chunk_retriever = chunk_vectorstore.as_retriever(
        search_kwargs={
            "k": 2  # Reduced from 3 to get more focused results
        }
    )
    
    page_retriever = page_vectorstore.as_retriever(
        search_kwargs={
            "k": 1  # Only get the most relevant page
        }
    )
    
    st.sidebar.success("✅ Vector Stores Created!")
except Exception as e:
    st.error(f"Error creating vector stores: {str(e)}")
    st.stop()

# ---------------------- Enhanced Prompt Template ----------------------
system_prompt = """You are a technical AI assistant specialized in providing precise answers about YOLOv9 research paper.

IMPORTANT GUIDELINES:
1. ONLY answer with information that DIRECTLY answers the user's question
2. If you find an exact quote that answers the question, use it and keep explanation minimal
3. Do not include tangential or contextual information unless specifically asked
4. If the exact answer isn't in the provided context, say "I cannot find a direct answer to this specific question in the paper"

Format your response as:

Direct Quote (Section X.X, Page Y):
"[exact quote that answers the question]"

Brief Explanation:
[Only if needed, provide a very brief explanation focusing strictly on what was asked]

Remember:
- Stay focused on the exact question asked
- Don't add unnecessary context or related information
- Be precise and concise
- Only use the most relevant quotes

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# ---------------------- Chat Processing ----------------------
query = st.chat_input("💬 Ask something about YOLOv9...")

if query:
    st.markdown(f'<div class="user-message">👤 **You:** {query}</div>', unsafe_allow_html=True)
    
    with st.spinner("🔍 Processing..."):
        try:
            # Get relevant chunks and pages with higher threshold
            relevant_chunks = chunk_retriever.get_relevant_documents(query)
            relevant_pages = page_retriever.get_relevant_documents(query)
            
            if not relevant_chunks and not relevant_pages:
                st.warning("No relevant information found.")
                st.stop()
            
            # Sort documents by relevance score if available
            if hasattr(relevant_chunks[0], 'metadata') and 'score' in relevant_chunks[0].metadata:
                relevant_chunks.sort(key=lambda x: x.metadata['score'], reverse=True)
            
            # Use only the most relevant chunk for the answer
            most_relevant_docs = relevant_chunks[:1]
            
            # Create and run RAG chain with focused documents
            qa_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt,
            )
            
            rag_chain = create_retrieval_chain(
                retriever=chunk_retriever,
                combine_docs_chain=qa_chain
            )
            
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]
            
            # Display response
            st.markdown(f'<div class="bot-message">🤖 **Gemini AI:** {answer}</div>', unsafe_allow_html=True)
            
            # Display only the most relevant page
            if relevant_pages:
                st.markdown("### 📚 Source Page")  # Changed from "Source Pages" to "Source Page"
                
                pdf_document = fitz.open(pdf_path)
                
                # Show only the most relevant page
                doc = relevant_pages[0]
                page_num = doc.metadata["page_number"]
                
                with st.container():
                    st.markdown(f"**Page {page_num}**")
                    
                    if doc.metadata.get("section"):
                        st.markdown(f"*Section: {doc.metadata['section']}*")
                    
                    st.markdown("**Relevant Text:**")
                    highlighted_text = highlight_text(doc.page_content, query)
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                    
                    try:
                        page = pdf_document[page_num - 1]
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        img_path = f"page_{page_num}.png"
                        pix.save(img_path)
                        st.image(img_path, caption=f"Page {page_num}", use_column_width=True)
                        os.remove(img_path)
                    except Exception as e:
                        st.error(f"Error displaying page {page_num}: {str(e)}")
                
                pdf_document.close()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
