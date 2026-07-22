# 🤖 RAG Chatbot - AI Document Assistant

A modern, professional RAG (Retrieval-Augmented Generation) chatbot that allows you to upload PDF documents and have intelligent conversations about their content.

## ✨ Features

- 📄 **PDF Upload & Processing** - Upload any PDF document for analysis
- 💬 **Intelligent Chat** - Ask questions and get accurate answers from your documents
- 🧠 **Conversation Memory** - Maintains context throughout the conversation
- 📚 **Structured Citations** - Every answer cites exact filename, page, chunk index, similarity score, and content preview
- ⛔ **Hallucination Prevention** - Similarity score threshold discards irrelevant chunks before LLM is called; explicit "No Relevant Context Found" response
- 🎨 **Modern UI** - Clean, professional interface with citation cards
- 🔄 **Session Management** - Multiple sessions with independent chat histories
- ⚡ **Fast Retrieval** - FAISS with `similarity_score_threshold` for relevance-gated retrieval
- 🌐 **RESTful API** - Well-documented API endpoints

## 🏗️ Architecture

```
rag-chatbot/
├── backend/
│   ├── main.py              # FastAPI application (lifespan pattern)
│   ├── config.py            # Configuration (threshold, temperature, chunk settings)
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── services/
│   │   ├── vector_service.py    # PDF ingestion, chunking, metadata enrichment, FAISS
│   │   └── chat_service.py      # RAG chain, hallucination prevention, chain caching
│   ├── models/
│   │   └── schemas.py       # Pydantic v2 models (Citation, ChatResponse)
│   └── utils/
│       └── helpers.py       # Utility functions
├── frontend/
│   ├── index.html          # Main HTML interface
│   ├── style.css           # Modern styling (citation cards, no-context badge)
│   └── script.js           # Frontend logic (structured citation renderer)
├── requirements.txt        # Python dependencies (LangChain 0.3.x)
├── EXAMPLES.md             # 3 example Q&A pairs with real pipeline output
└── README.md               # This file
```

### Pipeline Data Flow

```
[PDF Upload]
     │
     ▼
PyPDFLoader → raw pages
     │
     ▼
RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
     │
     ▼
Metadata enrichment per chunk:
  chunk_id | filename | page | chunk_index | total_chunks | char_count
     │
     ▼
OllamaEmbeddings (nomic-embed-text, local) → dense vectors
     │
     ▼
FAISS.from_documents() → saved to disk + in-memory cache

─────────────────────────────────────────────────────────────────────

[Chat Request]
     │
     ▼
history_aware_retriever:
  reformulate question using chat history (standalone question)
     │
     ▼
FAISS retriever  (search_type=similarity_score_threshold, k=4)
     │
     ├── 0 chunks passed threshold (score < 0.4)
     │       └─► Return "No Relevant Context Found" message
     │            (LLM never called — no hallucination possible)
     │
     └── 1-4 relevant chunks returned
             │
             ▼
         create_stuff_documents_chain
         + hardened system prompt (strict no-fabrication rules)
             │
             ▼
         ChatGroq (llama-3.3-70b-versatile, temperature=0.1)
             │
             ▼
         ChatResponse:
           answer | sources: List[Citation] | no_context_found
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Ollama** - For embeddings
3. **Groq API Key** - For LLM (get from [console.groq.com](https://console.groq.com))

### Installation

1. **Clone or download the project**
```bash
cd rag-chatbot
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install and setup Ollama**
```bash
# Install Ollama (if not installed)
# Visit: https://ollama.com/download

# Pull the embedding model
ollama pull nomic-embed-text

# Start Ollama server
ollama serve
```

4. **Configure environment variables**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional
OLLAMA_BASE_URL=http://localhost:11434
```

5. **Run the application**
```bash
# From the backend directory
cd backend
python main.py
```

6. **Access the application**
Open your browser and navigate to:
```
http://localhost:8000
```

API documentation available at:
```
http://localhost:8000/docs
```

## 📖 Usage

### Using the Web Interface

1. **Start a Session**
   - The app automatically generates a session ID

2. **Upload a PDF**
   - Click the upload area or drag & drop a PDF file
   - Wait for processing (you'll see pages and chunks count)

3. **Start Chatting**
   - Type your question in the input box
   - Press Enter to send
   - Get AI-powered answers with source citations

4. **Manage Sessions**
   - **Clear History**: Removes chat messages but keeps the document
   - **New Session**: Starts fresh with a new document

### Using the API

#### Health Check
```bash
curl http://localhost:8000/api/health
```

#### Upload PDF
```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@document.pdf" \
  -F "session_id=session_test123"
```

#### Send Chat Message
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is this document about?",
    "session_id": "session_test123"
  }'
```

#### Get Session Info
```bash
curl http://localhost:8000/api/sessions/session_test123/info
```

#### Clear Chat History
```bash
curl -X POST http://localhost:8000/api/sessions/session_test123/clear-history
```

#### Delete Session
```bash
curl -X DELETE http://localhost:8000/api/sessions/session_test123
```

## 🔧 Configuration

Edit `backend/config.py` to customize:

- **LLM Settings**: Model, temperature, max tokens
- **RAG Settings**: Chunk size, overlap, retrieval count
- **File Settings**: Max upload size, allowed extensions
- **Server Settings**: Host, port, CORS origins

## 🏢 Project Structure Details

### Backend Components

- **`main.py`**: FastAPI application setup and startup logic
- **`config.py`**: Centralized configuration using Pydantic Settings
- **`api/routes.py`**: All API endpoints with request/response handling
- **`services/vector_service.py`**: PDF processing and vector store management
- **`services/chat_service.py`**: RAG chain and conversation management
- **`models/schemas.py`**: Pydantic models for validation
- **`utils/helpers.py`**: Utility functions for file handling and validation

### Frontend Components

- **`index.html`**: Semantic HTML structure with accessibility features
- **`style.css`**: Modern CSS with CSS variables and responsive design
- **`script.js`**: Vanilla JavaScript for API interactions and UI updates

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern web framework
- **LangChain**: RAG orchestration
- **Groq**: Fast LLM inference (Llama 3)
- **Ollama**: Local embeddings (nomic-embed-text)
- **FAISS**: Efficient vector similarity search
- **PyPDF**: PDF text extraction

### Frontend
- **Vanilla JavaScript**: No framework dependencies
- **Modern CSS**: CSS Grid, Flexbox, Variables
- **Responsive Design**: Mobile-friendly interface

## 🐛 Troubleshooting

### PDF Upload Not Working

**Issue**: Upload fails or PDF processing errors

**Solutions**:
1. Check file size (must be < 10MB)
2. Ensure PDF is not password-protected
3. Verify PDF contains readable text (not scanned images)
4. Check backend logs for detailed errors

### Ollama Connection Issues

**Issue**: "Could not connect to Ollama" error

**Solutions**:
1. Verify Ollama is running: `ollama serve`
2. Check model is installed: `ollama list`
3. Pull model if missing: `ollama pull nomic-embed-text`
4. Verify `OLLAMA_BASE_URL` in `.env` file

### Chat Not Responding

**Issue**: Messages send but no response

**Solutions**:
1. Check Groq API key is valid
2. Verify session has uploaded PDF
3. Check backend logs for errors
4. Ensure internet connection for Groq API

### Port Already in Use

**Issue**: "Address already in use" error

**Solutions**:
1. Change port in `.env` file: `PORT=8001`
2. Or kill process using port 8000:
```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## 📝 API Documentation

Full API documentation is available at `/docs` when the server is running.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend interface |
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload PDF document |
| POST | `/api/chat` | Send chat message |
| GET | `/api/sessions/{id}/info` | Get session info |
| POST | `/api/sessions/{id}/clear-history` | Clear chat history |
| DELETE | `/api/sessions/{id}` | Delete session |
| GET | `/api/sessions` | List all sessions |

## 🔐 Security Notes

- Never commit `.env` file with real API keys
- Use environment variables for sensitive data
- Implement rate limiting for production
- Add authentication for multi-user environments
- Sanitize file uploads in production

## 🚀 Production Deployment

For production deployment:

1. **Set proper CORS origins** in `config.py`
2. **Use a production ASGI server**: Gunicorn with Uvicorn workers
3. **Add authentication** and rate limiting
4. **Use HTTPS** with proper SSL certificates
5. **Set up proper logging** and monitoring
6. **Configure firewall** rules
7. **Use a reverse proxy** (Nginx/Caddy)

Example production command:
```bash
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📄 License

This project is provided as-is for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Check backend logs for detailed errors

## 🔬 Improvements for Long Documents

*This write-up discusses architectural improvements for scaling the RAG pipeline
to handle long documents (100+ pages) with higher precision and recall.*

### Current limitations at scale

The current pipeline uses **fixed-size character chunking** (`RecursiveCharacterTextSplitter`,
chunk_size=1000). This approach works well for short documents but degrades on long ones:
a 300-page technical manual produces ~3,000 chunks, and the `k=4` retriever may miss
critical evidence spread across non-contiguous sections.

---

### 1. Semantic Chunking (replace fixed-size splitting)

Instead of splitting every 1,000 characters, **semantic chunking** groups sentences
by *meaning* using embedding similarity. LangChain experimental provides
`SemanticChunker`, which compares adjacent sentence embeddings and splits only
when semantic similarity drops below a threshold.

**Why it helps:** Logical units (paragraphs, arguments, table rows) stay together.
A section titled "Methodology" won't be split mid-sentence because a character
limit was hit. This improves retrieval precision because each chunk is a coherent
unit of meaning.

---

### 2. Hybrid Search (BM25 + Dense Retrieval)

FAISS performs **dense retrieval** — great for semantic similarity, but it misses
exact keyword matches (e.g., product codes, names, formulas). **BM25** is a
sparse, term-frequency-based method that excels at exact matches.

LangChain's `EnsembleRetriever` combines both:
- BM25 handles vocabulary (e.g., "Appendix B.3.1", "ISO-9001")
- FAISS handles paraphrase and semantic equivalence

Results from both retrievers are merged and normalized using Reciprocal Rank Fusion
(RRF). This hybrid approach consistently outperforms either method alone on
knowledge-intensive tasks.

---

### 3. Re-ranking with a Cross-Encoder

After retrieving `k=20` candidate chunks, a **cross-encoder** (e.g.,
`cross-encoder/ms-marco-MiniLM-L-6-v2` from Sentence Transformers) scores each
`(query, chunk)` pair jointly — not independently. Cross-encoders are more
accurate than bi-encoders because they see the query and chunk together.

The top 4 re-ranked chunks are passed to the LLM. This two-stage pipeline
(retrieve broadly → re-rank precisely) dramatically reduces false positives
and improves answer quality on long documents where many chunks superficially
match the query.

---

### 4. Parent-Document Retriever (multi-vector)

Store **small child chunks** for precision retrieval, but return **large parent
chunks** to the LLM for richer context. LangChain's `ParentDocumentRetriever`
implements this pattern:
- Child chunks (256 tokens) are embedded and retrieved by FAISS
- Their parent chunks (1,024 tokens) are returned to the LLM

This solves the "lost in the middle" problem where critical context is adjacent
to but not inside the retrieved chunk.

---

### 5. Late Chunking / Token-level Retrieval

**Late chunking** (ColBERT-style) embeds the full document in one pass, then
produces per-token embeddings. At query time, token-level maximum similarity
(MaxSim) is computed between query tokens and document tokens.

This captures fine-grained relevance that chunk-level embeddings miss — especially
valuable for long technical documents where a single paragraph contains answers
to multiple different questions.

---

### Summary Table

| Technique | Addresses | Complexity | LangChain Support |
|---|---|---|---|
| Semantic Chunking | Poor chunk boundaries | Low | `SemanticChunker` (experimental) |
| Hybrid Search (BM25 + FAISS) | Vocabulary mismatch | Medium | `EnsembleRetriever` |
| Cross-encoder Re-ranking | Retrieval false positives | Medium | Custom with `sentence-transformers` |
| Parent-Document Retriever | Lost context adjacency | Medium | `ParentDocumentRetriever` |
| Late Chunking (ColBERT) | Token-level precision | High | External (PyLate / RAGatouille) |

---

## 🎯 Future Enhancements

- [ ] Support for multiple file formats (DOCX, TXT, etc.)
- [ ] User authentication and authorization
- [ ] Conversation export (PDF, JSON)
- [ ] Hybrid search (BM25 + FAISS `EnsembleRetriever`)
- [ ] Cross-encoder re-ranking for long documents
- [ ] Real-time collaboration
- [ ] Cloud storage integration
- [ ] Multiple language support

---

**Built with ❤️ using FastAPI, LangChain 0.3.x, and modern web technologies**
