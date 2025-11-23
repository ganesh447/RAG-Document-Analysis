"""
FastAPI backend for RAG Document Summarizer
Handles file uploads, URL processing, and query processing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import os
import tempfile
import shutil
import uuid
from pathlib import Path
from gtts import gTTS
import io

from main import (
    RAGPipeline, 
    load_document, 
    load_website,
    build_retriever,
    HuggingFaceEmbeddings,
    RecursiveCharacterTextSplitter
)

app = FastAPI(title="RAG Document Summarizer API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:5173", "http://127.0.0.1:8080"],  # Vite ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for active RAG pipelines (in production, use proper session management)
active_pipelines = {}

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    llm_model: str = "mistral"
    embedding_model: str = "all-MiniLM-L6-v2"
    tone: str = "neutral"
    top_k: int = 5

class QueryResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    context_snippets: Optional[List[str]] = None
    message: Optional[str] = None

class URLProcessRequest(BaseModel):
    url: str
    llm_model: str = "mistral"
    embedding_model: str = "all-MiniLM-L6-v2"

class URLProcessResponse(BaseModel):
    status: str
    session_id: Optional[str] = None
    message: Optional[str] = None

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# Available embedding models - single source of truth
AVAILABLE_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2",
    "nomic-embed-text": "nomic-embed-text"
}

# Available LLM models
AVAILABLE_LLM_MODELS = ["mistral", "llava"]

def get_embedding_model(model_name: str):
    """Get embedding model by name"""
    if model_name not in AVAILABLE_EMBEDDING_MODELS:
        raise ValueError(f"Embedding model '{model_name}' not supported. Available: {list(AVAILABLE_EMBEDDING_MODELS.keys())}")
    
    return HuggingFaceEmbeddings(model_name=AVAILABLE_EMBEDDING_MODELS[model_name])

@app.get("/")
async def root():
    return {"message": "RAG Document Summarizer API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/models")
async def get_available_models():
    """
    Get list of available LLM and embedding models
    """
    return {
        "llm_models": AVAILABLE_LLM_MODELS,
        "embedding_models": list(AVAILABLE_EMBEDDING_MODELS.keys())
    }

@app.post("/upload", response_model=dict)
async def upload_file(
    file: UploadFile = File(...),
    llm_model: str = Form("mistral"),
    embedding_model: str = Form("all-MiniLM-L6-v2")
):
    """
    Upload a PDF/document file and create a RAG pipeline session
    """
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.docx', '.txt')):
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF, DOCX, and TXT files are supported.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Load document with better error handling
            try:
                docs = load_document(tmp_path)
            except Exception as doc_error:
                error_msg = str(doc_error)
                # Check for specific PDF errors
                if "404" in error_msg or "Cannot locate document" in error_msg:
                    raise HTTPException(
                        status_code=400, 
                        detail="The PDF file appears to be corrupted, password-protected, or invalid. Please try a different file."
                    )
                elif "password" in error_msg.lower() or "encrypted" in error_msg.lower():
                    raise HTTPException(
                        status_code=400,
                        detail="The PDF file is password-protected. Please remove the password and try again."
                    )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error reading document: {error_msg}"
                    )
            
            # Create embedding model
            embedding = get_embedding_model(embedding_model)
            
            # Build FAISS index
            chunks = splitter.split_documents(docs)
            vectordb = build_faiss_index_with_embedding(chunks, embedding)
            retriever = build_retriever(vectordb)
            
            # Create RAG pipeline with the selected embedding model
            rag = RAGPipeline(model_name=llm_model, embedding_model_name=embedding_model)
            rag.vectordb = vectordb
            rag.retriever = retriever
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            active_pipelines[session_id] = rag
            
            return {
                "status": "success",
                "session_id": session_id,
                "filename": file.filename,
                "message": "File uploaded and processed successfully"
            }
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper status codes)
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/process-url", response_model=URLProcessResponse)
async def process_url(request: URLProcessRequest):
    """
    Process a website URL and create a RAG pipeline session
    """
    try:
        # Validate URL
        if not request.url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Invalid URL. Must start with http:// or https://")
        
        # Load website content
        docs = load_website(request.url)
        
        # Create embedding model
        embedding = get_embedding_model(request.embedding_model)
        
        # Build FAISS index
        chunks = splitter.split_documents(docs)
        vectordb = build_faiss_index_with_embedding(chunks, embedding)
        retriever = build_retriever(vectordb)
        
        # Create RAG pipeline with the selected embedding model
        rag = RAGPipeline(model_name=request.llm_model, embedding_model_name=request.embedding_model)
        rag.vectordb = vectordb
        rag.retriever = retriever
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        active_pipelines[session_id] = rag
        
        return URLProcessResponse(
            status="success",
            session_id=session_id,
            message="URL processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")

@app.post("/query/{session_id}", response_model=QueryResponse)
async def query_document(
    session_id: str,
    request: QueryRequest
):
    """
    Query a document using an existing session
    """
    if session_id not in active_pipelines:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a file or process a URL first.")
    
    try:
        rag = active_pipelines[session_id]
        
        # Retrieve relevant chunks
        chunks = rag.retrieve_chunks(request.question, request.top_k)
        context_snippets = [chunk.page_content for chunk in chunks]
        
        # Generate answer
        answer = rag.generate_answer(request.question, request.tone, request.top_k)
        
        return QueryResponse(
            status="success",
            answer=answer,
            context_snippets=context_snippets
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and clean up resources
    """
    if session_id in active_pipelines:
        del active_pipelines[session_id]
        return {"status": "success", "message": "Session deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"  # Language code (e.g., 'en', 'es', 'fr', etc.)
    slow: bool = False  # Slow down speech

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using gTTS
    Returns an MP3 audio file
    """
    try:
        # Limit text length to prevent abuse (gTTS has limits)
        if len(request.text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long. Maximum 5000 characters.")
        
        # Create gTTS object
        tts = gTTS(text=request.text, lang=request.lang, slow=request.slow)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        # Return the audio file with cleanup
        def cleanup():
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
        
        return FileResponse(
            tmp_path,
            media_type="audio/mpeg",
            filename="speech.mp3",
            background=cleanup
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

def build_faiss_index_with_embedding(docs, embedding_model):
    """Build FAISS index with a specific embedding model"""
    from langchain_community.vectorstores import FAISS
    return FAISS.from_documents(docs, embedding_model)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

