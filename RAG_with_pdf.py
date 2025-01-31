# Import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()  

# Step 1: Load the PDF file
# This will load the entire document as a single text block.
loader = PyPDFLoader("yolov9_paper.pdf")
data = loader.load()  # Load the entire PDF as a document

# Step 2: Split the document into smaller chunks for better processing
# RecursiveCharacterTextSplitter ensures we maintain logical splits within the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)  # Each chunk is 1000 characters
docs = text_splitter.split_documents(data)

print("Total number of documents after splitting: ", len(docs))  # Display the number of chunks

# Print the 8th document chunk for reference (index 7, as lists start from 0)
print(docs[7])


# Step 3: Create an embedding model using Google Generative AI
# Embeddings help convert text into numerical representations for vector search
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Test the embedding model with a small text sample
vector = embeddings.embed_query("hello, world!")
print("Sample embedding vector (first 5 values):", vector[:5])  # Show part of the generated vector


# Step 4: Store document embeddings in a vector database (ChromaDB)
# ChromaDB allows us to efficiently search and retrieve similar text
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

print("Vectorstore initialized successfully!")

# Step 5: Create a retriever from the vector database
# This retriever finds the most relevant document chunks based on the user's query
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})  # Retrieve top 10 matches

# Step 6: Test the retriever with a query about YOLOv9
retrieved_docs = retriever.invoke("What is new in YOLOv9?")

print("Number of retrieved documents:", len(retrieved_docs))  # Display how many documents were retrieved

# Display content from one of the retrieved documents (5th one)
print("Sample retrieved document:", retrieved_docs[5].page_content)


# Step 7: Initialize the language model (Gemini AI) for answering questions
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

# Step 8: Define a prompt template for RAG-based Q&A
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Step 9: Create a structured prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # System message defining assistant behavior
        ("human", "{input}"),       # User input placeholder
    ]
)

# Step 10: Create a retrieval-based question-answering chain (RAG)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Step 11: Ask a question about YOLOv9 and get an answer
query = "What is new in YOLOv9?"
response = rag_chain.invoke({"input": query})

# Step 12: Print the generated answer
print("AI's Answer:", response["answer"])
