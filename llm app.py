import os
import unicodedata
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login

# Access keys securely using Streamlit Secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
huggingface_token = st.secrets["HUGGINGFACE_TOKEN"]

# Log in to Hugging Face
login(token=huggingface_token)

# Function to clean text by removing invalid Unicode characters
def clean_text(text):
    """Removes invalid characters from the text."""
    return ''.join(c for c in text if not unicodedata.combining(c))

# Function to get text from a website using WebBaseLoader
def get_website_text(url):
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    text = ""
    for doc in docs:
        text += doc.page_content
    return text

# Function to get text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create the vector store using selected embeddings
def get_vector_store(text_chunks, embedding_choice):
    if embedding_choice == "Google Generative AI Embeddings":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")
    elif embedding_choice == "all-MiniLM-L6-v2":
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

# Function to create the conversational chain based on model selection
def get_conversational_chain(model_choice):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
    just say "answer is not available in the context" and do not guess.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """

    if model_choice == "Gemini":
        model = ChatGoogleGenerativeAI(api_key=os.environ["GOOGLE_API_KEY"], model="gemini-pro")
    elif model_choice == "Gemma":
        model = HuggingFaceEndpoint(repo_id="google/gemma-1.1-2b-it", max_length=1024, temperature=0.1)
    elif model_choice == "OpenAI":
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if openai_api_key:
            model = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to get the response based on user input question
def user_input(user_question, text_chunks, model_choice, embedding_choice):
    # Get the vector store in memory
    vector_store = get_vector_store(text_chunks, embedding_choice)

    # Use the vector store to get relevant documents based on the user question
    docs = vector_store.similarity_search(user_question)

    # Get the conversational chain
    chain = get_conversational_chain(model_choice)

    # Get the response
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

# Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF or Website", layout="wide")
    st.title("Chat with PDF or Website Content 📚💬")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for file upload or website URL input
    with st.sidebar:
        st.header("Upload PDF or Provide Website URL")
        pdf_docs = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)
        website_url = st.text_input("Or Enter Website URL", "")
        
        # Model selection
        model_choice = st.selectbox("Choose a language model:", ["Gemini", "Gemma", "OpenAI"])
        embedding_choice = st.selectbox("Choose an embedding model:", 
                                        ["Google Generative AI Embeddings", "all-MiniLM-L6-v2", 
                                         "distilbert-base-nli-stsb-mean-tokens", "msmarco-distilbert-base-v4"])

        # Ask question from the user
        user_question = st.text_input("Ask a question from the content")

    # Check if a PDF is uploaded or a website URL is provided
    if pdf_docs:
        raw_text = clean_text(get_pdf_text(pdf_docs))  # Clean text after extracting from PDF
    elif website_url:
        raw_text = clean_text(get_website_text(website_url))  # Clean text after extracting from website
    else:
        st.warning("Please upload a PDF file or enter a website URL.")
        return

    # Split the text into chunks
    text_chunks = get_text_chunks(raw_text)

    # Process user question if provided
    if user_question:
        st.write(f"Question: {user_question}")
        answer = user_input(user_question, text_chunks, model_choice, embedding_choice)
        st.write(f"Answer: {answer}")

        # Append to chat history
        st.session_state.chat_history.append({"user": user_question, "agent": answer})

    # Display Chat History
    st.header("Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"You: {chat['user']}")
        st.write(f"Agent: {chat['agent']}")

if __name__ == "__main__":
    main()
