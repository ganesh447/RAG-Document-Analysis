# RAG Document Summarizer

A full-stack application for querying documents and websites using RAG (Retrieval-Augmented Generation) with local LLM models via Ollama.

## Features

- **Document Processing**: Upload and query PDF, DOCX, and TXT files
- **Website Processing**: Scrape and query website content
- **Multiple LLM Models**: Support for Mistral and Llava models
- **Multiple Embedding Models**: Choose from different embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, nomic-embed-text)
- **Interactive Frontend**: Modern React-based UI with real-time querying

## Prerequisites

1. **Python 3.8+**
2. **Node.js 18+** (for frontend)
3. **Ollama** installed and running with models:
   - `mistral` (or your preferred text model)
   - `llava` (for vision tasks, if needed)

### Installing Ollama Models

```bash
# Install Mistral
ollama pull mistral

# Install Llava (optional, for vision tasks)
ollama pull llava
```

## Installation

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd smart-document-answer
```

2. Install dependencies:
```bash
npm install
```

## Running the Application

### 1. Start the Backend API

From the root directory:
```bash
python api.py
```

The API will run on `http://localhost:8000`

### 2. Start the Frontend

From the `smart-document-answer` directory:
```bash
npm run dev
```

The frontend will run on `http://localhost:8080` (or the port shown in the terminal)

### 3. Access the Application

Open your browser and navigate to `http://localhost:8080`

## API Endpoints

### Upload Document
- **POST** `/upload`
- **Body**: Form data with `file`, `llm_model`, `embedding_model`
- **Response**: Session ID for querying

### Process URL
- **POST** `/process-url`
- **Body**: JSON with `url`, `llm_model`, `embedding_model`
- **Response**: Session ID for querying

### Query Document
- **POST** `/query/{session_id}`
- **Body**: JSON with `question`, `llm_model`, `embedding_model`, `tone`, `top_k`
- **Response**: Answer and context snippets

### Delete Session
- **DELETE** `/session/{session_id}`
- **Response**: Success message

## Usage

1. **Select Source Type**: Choose between PDF upload or Website URL
2. **Select Models**: Choose your preferred LLM model (Mistral/Llava) and embedding model
3. **Upload/Process**: Upload a document or enter a website URL and click "Process"
4. **Ask Questions**: Once processed, enter your question and click "Ask Question"
5. **View Results**: See the AI-generated answer along with relevant context snippets

## Project Structure

```
RAG-Scraper/
├── api.py                 # FastAPI backend
├── main.py                # Core RAG pipeline logic
├── requirements.txt       # Python dependencies
├── smart-document-answer/ # React frontend
│   ├── src/
│   │   ├── pages/
│   │   │   └── Index.tsx  # Main application page
│   │   └── ...
│   └── ...
└── README.md
```

## Notes

- The backend uses in-memory session storage. Sessions are lost when the server restarts.
- For production, consider implementing proper session management (Redis, database, etc.)
- Make sure Ollama is running before starting the backend
- Large documents may take time to process

## Troubleshooting

### Backend Issues
- Ensure Ollama is running: `ollama list`
- Check if models are installed: `ollama list`
- Verify Python dependencies: `pip list`

### Frontend Issues
- Clear browser cache
- Check browser console for errors
- Verify API URL in frontend code matches backend port

### CORS Issues
- If frontend runs on a different port, update CORS origins in `api.py`

## License

MIT

