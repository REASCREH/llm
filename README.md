## 🌟 Chat with PDFs or Websites 📚💬


Transform the way you interact with information! This Streamlit application enables you to upload PDFs or input website URLs and query their content using state-of-the-art AI models. Whether you're analyzing research, reviewing documents, or exploring online resources, this app delivers accurate, insightful responses tailored to your questions.


🚀 Live App

Try https://chatwebpdf.streamlit.app/

📂 GitHub Repository
GitHub https://github.com/REASCREH/llm

Overview

This Python script, built using the Streamlit framework, creates a web application that allows users to interact with PDF documents or web content through a conversational interface.

Key Functionalities:

Document Input:

Users can either upload a PDF file or provide a website URL.

Text Extraction:

The script extracts text from PDF documents using PyPDF2 and from websites using WebBaseLoader.
Extracted text is cleaned to remove invalid Unicode characters.

Text Chunking:

The extracted text is split into smaller chunks using RecursiveCharacterTextSplitter for efficient processing.
Vector Store Creation:

The text chunks are embedded using either Google Generative AI Embeddings or Hugging Face Embeddings.
These embeddings are then used to create a vector store using FAISS, which enables semantic search.

Conversational Interface:

Users can input questions related to the document content.
The vector store is queried to retrieve relevant context.
The selected language model (Gemini, Gemma, or OpenAI) is used to generate a response based on the query and retrieved context.

Chat History:

The application maintains a chat history, allowing users to review past interactions.










## Features of the "Chat with PDF or Website Content" Application


This application offers powerful features for extracting, analyzing, and interacting with text from PDFs or websites using advanced NLP capabilities.

1. Multi-Source Content Processing

PDF Support: Upload and process multiple PDF files simultaneously.
Website Integration: Extract text content directly from any website using its URL.

2. Advanced NLP Models

Model Selection: Choose from multiple NLP models, including:
Gemini (Google Generative AI).
Gemma (HuggingFace).
OpenAI GPT (e.g., GPT-3.5-turbo).
Customizable Prompts: Tailored prompts for accurate and detailed question answering.

3. Embedding Options for Text Representation


Embedding Models: Select from various embedding models for context-aware responses:
Google Generative AI Embeddings.

HuggingFace (e.g., all-MiniLM-L6-v2).
Other options like distilbert-base-nli-stsb-mean-tokens and msmarco-distilbert-base-v4.

4. Efficient Text Splitting and Chunking

Recursive Text Splitting: Break large documents into manageable text chunks.
Overlap Handling: Includes overlapping content between chunks for better context.

5. Interactive Conversational Chain

Question-Answering Chain: Provides contextually accurate answers based on the content.
Error Handling: Responds with "answer not available in the context" when necessary to avoid guesswork.

6. Chat History Management

Session State Memory: Maintains a history of user queries and responses.
Easy Reference: Users can review past interactions during the same session.

7. Flexible User Interaction

User-Friendly Interface: Powered by Streamlit for an intuitive and interactive experience.
Sidebar Navigation: Simplifies file uploads, URL input, and model selection.
Secure Key Input: Allows secure entry of API keys for OpenAI or Google services.

8. Privacy and Security

Local Processing: Works locally, ensuring sensitive documents and queries remain private.
API Key Control: Users can provide their own keys for external model usage.





## ```Advantages
Time Efficiency:

(1) Quickly extract and retrieve specific information from lengthy documents or web content.

(2) No need to manually search through pages; ask a question and get precise answers.

Versatile Input Options:


(3) Supports both PDF uploads and website URLs, making it flexible for different types of content sources.
Enhanced Accessibility:


(4) Simplifies complex documents by breaking them into manageable chunks and enabling natural language queries.
(5) Helps users with limited technical expertise interact with advanced NLP models through an intuitive interface.

Customizable NLP Models:

(6) Offers multiple language models like Gemini, Gemma, and OpenAI GPT for tailored use cases.
(7) Users can choose the embedding model that best fits their needs, such as Google Generative AI or HuggingFace.

Improved Accuracy:


(8) Uses embeddings and vector stores for similarity searches, ensuring contextually relevant responses.
(9) The application is designed to avoid guesswork by only answering based on the provided context.

Knowledge Retention:


(10) Maintains chat history, allowing users to refer back to previous queries and responses.
(11) Useful for ongoing projects or when reviewing multiple questions over time.

Wide Range of Use Cases:

(12) Ideal for various domains, including education, research, business, healthcare, legal, and market analysis.
(13) Adaptable for personal, academic, or professional applications.

User-Friendly Interface:

(14) Streamlit ensures a smooth and interactive experience with no steep learning curve.
(15) Sidebar navigation and chat-based interaction enhance usability.

Data Privacy:


(16) Works locally on the user’s machine, ensuring that sensitive documents and queries remain private.
(17) Offers control over API keys for external models like OpenAI or Google Generative AI.

Cost-Effective:


(18) Open-source and customizable, reducing dependency on expensive enterprise solutions.
(19) Allows users to select free or low-cost models and embeddings like HuggingFace for cost management.

Scalability:


(20) The modular architecture allows easy integration of additional models, embeddings, or features as needed.
(21) Potential for cloud integration to handle larger workloads or collaborative environments.

AI-Driven Insights:


(22) Extracts insights and generates contextual answers from vast amounts of text.
(23) Supports decision-making processes by providing concise and relevant information.











## Getting Started





## To run this application locally, follow these steps:



Clone the project

```bash
  git clone https://github.com/REASCREH/llm.git




```

Go to the project directory

```bash
  cd llm


```

Install dependencies

```bash
  pip install -r requirements.txt
  GOOGLE_API_KEY="your_google_api_key_here"
HUGGINGFACE_TOKEN =""
OPENAI_API_KEY="your_opan_ai_key"

```


Start the server

```bash
  streamlit llm_app.py


```


Usage

Upload Data: After starting the app, upload your CSV, Excel, or SQL file.

Ask Questions: Enter questions about your data in the input box.

View Responses: The AI agent will analyze the data and provide answers.

Export to PDF: Download your chat history as a PDF for documentation or sharing purposes.

## How It Works


The "Chat with PDF or Website Content" application operates in a series of streamlined steps, ensuring an efficient and user-friendly experience:

1. Input Options

PDF Upload: Users can upload one or more PDF files directly.
Website URL: Alternatively, users can input a website URL for scraping content.

2. Content Extraction

For PDFs, the text is extracted using PyPDF2.
For websites, the content is fetched using WebBaseLoader.

3. Text Cleaning and Chunking

The extracted content is cleaned of invalid characters using a custom cleaning function.
The cleaned text is divided into manageable chunks using RecursiveCharacterTextSplitter for efficient processing.

4. Vector Store Creation

Users select an embedding model (e.g., Google Generative AI Embeddings or HuggingFace Embeddings) to encode the text.
The chunks are stored in a vector database (FAISS) for similarity searches.

5. Model Selection

Users choose a conversational model from options like Gemini, Gemma, or OpenAI GPT.
Each model is optimized to answer questions based on provided context without making guesses.

6. User Query and Response

Users input a question via the Streamlit interface.
Relevant text chunks are retrieved from the vector store based on similarity to the query.
The chosen model processes the query along with retrieved text and generates a context-aware response.

7. Chat History

The application maintains a history of all user queries and responses, allowing users to refer back to previous interactions easily.
