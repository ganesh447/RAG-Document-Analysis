# -----------------------------
# Install dependencies if not done
# -----------------------------
# pip install faiss-cpu sentence-transformers beautifulsoup4 requests python-docx pypdf ollama

import ollama
import os
from bs4 import BeautifulSoup
import requests

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# -----------------------------
# 1. Prompt Template
# -----------------------------

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs):
        formatted = self.template
        for key, value in kwargs.items():
            formatted = formatted.replace("{" + key + "}", value)
        return formatted


# Default template
DEFAULT_TEMPLATE = PromptTemplate("""
You are an assistant that answers based ONLY on the given context.

Context:
{context}

Question:
{question}

Tone: {tone}

Answer:
""")


# -----------------------------
# 2. Document / Website Loaders
# -----------------------------

def load_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            raise Exception("Unsupported file format.")
        
        docs = loader.load()
        
        # Check if document is empty
        if not docs or len(docs) == 0:
            raise ValueError("The document appears to be empty or could not be read.")
        
        return docs
    except Exception as e:
        # Re-raise with more context
        raise Exception(f"Failed to load document '{file_path}': {str(e)}")

def load_website(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    return [Document(page_content=text, metadata={"source": url})]  # same format as document loader


# -----------------------------
# 3. Text Splitter
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# -----------------------------
# 4. FAISS + Embeddings
# -----------------------------

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_faiss_index(docs, embedding_model_instance=None):
    """
    Build FAISS index with optional embedding model
    
    Args:
        docs: List of documents
        embedding_model_instance: Optional embedding model instance. If None, uses default.
    """
    if embedding_model_instance is None:
        embedding_model_instance = embedding_model
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, embedding_model_instance)
    return vectordb


# -----------------------------
# 5. Retriever
# -----------------------------

def build_retriever(vectordb, search_type="mmr", k=5, lambda_mult=0.5):
    return vectordb.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "lambda_mult": lambda_mult}
    )


# -----------------------------
# 6. RAG Pipeline Using Ollama
# -----------------------------

class RAGPipeline:
    def __init__(self, model_name="mistral", prompt_template: PromptTemplate = DEFAULT_TEMPLATE, embedding_model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.embedding_model_name = embedding_model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectordb = None
        self.retriever = None

    def add_documents(self, docs):
        """
        Build FAISS index and retriever from document chunks
        """
        self.vectordb = build_faiss_index(docs, self.embedding_model)
        self.retriever = build_retriever(self.vectordb)

    def retrieve_chunks(self, query: str, top_k: int = 5):
        if not self.retriever:
            raise ValueError("Retriever not initialized. Add documents first.")
        return self.retriever.invoke(query)[:top_k]

    def generate_answer(self, question: str, tone: str = "neutral", top_k: int = 5):
        """
        Generate answer using Ollama + retrieved chunks + prompt template
        """
        chunks = self.retrieve_chunks(question, top_k)
        context = "\n\n".join([c.page_content for c in chunks])

        # Fill template variables
        prompt = self.prompt_template.format(
            context=context,
            question=question,
            tone=tone
        )

        messages = [
            {"role": "system", "content": "Follow the user's custom prompt exactly."},
            {"role": "user", "content": prompt}
        ]

        response = ollama.chat(model=self.model_name, messages=messages)
        return response["message"]["content"]


# -----------------------------
# 7. Convenience Wrappers (Frontend-ready functions)
# -----------------------------

def ask_from_file(file_path, question, tone="neutral", model_name="mistral", embedding_model_name="all-MiniLM-L6-v2"):
    """
    Process a document file and answer a question.
    
    Args:
        file_path: Path to the document (PDF, DOCX, or TXT)
        question: User's question about the document
        tone: Response tone (neutral, simple, professional, etc.)
        model_name: Ollama model name to use
        embedding_model_name: Embedding model name to use
    
    Returns:
        str: Generated answer
    """
    docs = load_document(file_path)
    rag = RAGPipeline(model_name=model_name, embedding_model_name=embedding_model_name)
    rag.add_documents(docs)
    return rag.generate_answer(question, tone)

def ask_from_url(url, question, tone="neutral", model_name="mistral", embedding_model_name="all-MiniLM-L6-v2"):
    """
    Process a website URL and answer a question.
    
    Args:
        url: Website URL to scrape
        question: User's question about the website content
        tone: Response tone (neutral, simple, professional, etc.)
        model_name: Ollama model name to use
        embedding_model_name: Embedding model name to use
    
    Returns:
        str: Generated answer
    """
    docs = load_website(url)
    rag = RAGPipeline(model_name=model_name, embedding_model_name=embedding_model_name)
    rag.add_documents(docs)
    return rag.generate_answer(question, tone)

def process_query(source_type, source_path, question, tone="neutral", model_name="mistral", embedding_model_name="all-MiniLM-L6-v2"):
    """
    Unified function to process queries from either PDF or website.
    Frontend-ready function that can be called from API endpoints.
    
    Args:
        source_type: "pdf" or "website"
        source_path: File path (for PDF) or URL (for website)
        question: User's question
        tone: Response tone
        model_name: Ollama model name
        embedding_model_name: Embedding model name
    
    Returns:
        dict: Contains 'answer' and 'status'
    """
    try:
        if source_type.lower() == "pdf" or source_type.lower() == "file":
            answer = ask_from_file(source_path, question, tone, model_name, embedding_model_name)
        elif source_type.lower() == "website" or source_type.lower() == "url":
            answer = ask_from_url(source_path, question, tone, model_name, embedding_model_name)
        else:
            return {
                "status": "error",
                "answer": None,
                "message": f"Invalid source_type: {source_type}. Use 'pdf' or 'website'."
            }
        
        return {
            "status": "success",
            "answer": answer,
            "source_type": source_type,
            "source_path": source_path,
            "question": question
        }
    except Exception as e:
        return {
            "status": "error",
            "answer": None,
            "message": str(e)
        }


# -----------------------------
# 8. Interactive CLI (for testing)
# -----------------------------

def interactive_mode():
    """
    Interactive command-line interface for testing the RAG system.
    """
    print("=" * 60)
    print("RAG Document Summarizer - Interactive Mode")
    print("=" * 60)
    print()
    
    # Source type selection
    print("Select source type:")
    print("1. PDF/Document File")
    print("2. Website URL")
    source_choice = input("Enter choice (1 or 2): ").strip()
    
    if source_choice == "1":
        source_type = "pdf"
        source_path = input("Enter file path: ").strip()
        if not os.path.exists(source_path):
            print(f"Error: File '{source_path}' not found!")
            return
    elif source_choice == "2":
        source_type = "website"
        source_path = input("Enter website URL: ").strip()
        if not source_path.startswith(("http://", "https://")):
            print("Warning: URL should start with http:// or https://")
    else:
        print("Invalid choice!")
        return
    
    # Model selection
    model_name = input("Enter Ollama model name (default: mistral): ").strip() or "mistral"
    
    # Tone selection
    print("\nSelect response tone:")
    print("1. Neutral (default)")
    print("2. Simple")
    print("3. Professional")
    print("4. Casual")
    tone_choice = input("Enter choice (1-4, default: 1): ").strip()
    tone_map = {"1": "neutral", "2": "simple", "3": "professional", "4": "casual"}
    tone = tone_map.get(tone_choice, "neutral")
    
    print("\n" + "=" * 60)
    print("Processing document...")
    print("=" * 60)
    
    # Process the document
    try:
        if source_type == "pdf":
            docs = load_document(source_path)
        else:
            docs = load_website(source_path)
        
        rag = RAGPipeline(model_name=model_name)
        rag.add_documents(docs)
        print("Document processed successfully!")
        print()
    except Exception as e:
        print(f"Error processing document: {e}")
        return
    
    # Interactive Q&A loop
    print("=" * 60)
    print("Enter your questions (type 'quit' or 'exit' to stop)")
    print("=" * 60)
    print()
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        print("\nGenerating answer...")
        try:
            answer = rag.generate_answer(question, tone)
            print("\n" + "-" * 60)
            print("Answer:")
            print("-" * 60)
            print(answer)
            print("-" * 60)
            print()
        except Exception as e:
            print(f"Error generating answer: {e}")
            print()


if __name__ == "__main__":
    # Run interactive mode
    interactive_mode()
    
    # Example usage of the frontend-ready function:
    # result = process_query("pdf", "Sai_Ganesh_CV.pdf", "Summarize the methodology", "simple", "mistral")
    # print(result)
