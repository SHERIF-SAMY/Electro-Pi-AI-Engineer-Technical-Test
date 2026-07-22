"""
Vector store service for PDF processing and retrieval (Multi-document support)
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import shutil

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

from backend.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector stores"""
    
    def __init__(self):
        """Initialize the vector store service"""
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        # Cache is now session_id -> dict of filename -> FAISS
        self.vectorstore_cache: Dict[str, Dict[str, FAISS]] = {}
        self.session_metadata: Dict[str, dict] = {}
        
        logger.info(f"VectorStoreService initialized with model: {settings.EMBEDDING_MODEL}")
    
    def _sanitize_filename(self, filename: str) -> str:
        return filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    async def process_pdf(self, pdf_path: str, session_id: str, original_filename: str) -> Tuple[int, int]:
        """Process PDF file and create vector store"""
        try:
            # Load PDF
            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("PDF contains no readable text")
            
            num_pages = len(documents)
            logger.info(f"Loaded {num_pages} pages from PDF")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            num_chunks = len(splits)
            logger.info(f"Split into {num_chunks} chunks")
            
            # Enrich each chunk with traceable metadata
            for idx, doc in enumerate(splits):
                doc.metadata["chunk_id"] = f"{session_id}_{original_filename}_p{doc.metadata.get('page', 0)}_c{idx}"
                doc.metadata["chunk_index"] = idx
                doc.metadata["total_chunks"] = num_chunks
                doc.metadata["filename"] = original_filename
                doc.metadata["char_count"] = len(doc.page_content)
            logger.info(f"Enriched {num_chunks} chunks with metadata")
            
            # Create vectorstore with embeddings
            logger.info("Creating embeddings...")
            vectorstore = FAISS.from_documents(splits, self.embeddings)
            
            # Save to disk
            session_dir = settings.VECTORS_DIR / session_id
            session_dir.mkdir(exist_ok=True)
            
            safe_filename = self._sanitize_filename(original_filename)
            vectorstore_path = session_dir / safe_filename
            vectorstore.save_local(str(vectorstore_path))
            
            # Cache in memory
            if session_id not in self.vectorstore_cache:
                self.vectorstore_cache[session_id] = {}
            self.vectorstore_cache[session_id][safe_filename] = vectorstore
            
            # Store metadata
            if session_id not in self.session_metadata:
                self.session_metadata[session_id] = {"documents": {}}
                
            self.session_metadata[session_id]["documents"][safe_filename] = {
                "pages": num_pages,
                "chunks": num_chunks,
                "original_filename": original_filename
            }
            
            logger.info(f"Successfully processed PDF for session {session_id}, file {original_filename}")
            
            return num_pages, num_chunks
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def get_vectorstore(self, session_id: str, filename: str) -> FAISS:
        """Get or load vectorstore for a session and specific filename"""
        safe_filename = self._sanitize_filename(filename)
        
        # Check cache first
        if session_id in self.vectorstore_cache and safe_filename in self.vectorstore_cache[session_id]:
            logger.info(f"Retrieved vectorstore from cache for session {session_id}, file {filename}")
            return self.vectorstore_cache[session_id][safe_filename]
        
        # Try loading from disk
        vectorstore_path = settings.VECTORS_DIR / session_id / safe_filename
        
        if not vectorstore_path.exists():
            raise ValueError(f"No vectorstore found for session {session_id}, file {filename}")
        
        try:
            logger.info(f"Loading vectorstore from disk for session {session_id}, file {filename}")
            vectorstore = FAISS.load_local(
                str(vectorstore_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Cache for future use
            if session_id not in self.vectorstore_cache:
                self.vectorstore_cache[session_id] = {}
            self.vectorstore_cache[session_id][safe_filename] = vectorstore
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to load vectorstore: {str(e)}")
            raise ValueError(f"Failed to load vectorstore: {str(e)}")
            
    def get_all_vectorstores(self, session_id: str) -> Dict[str, FAISS]:
        """Get all vectorstores for a session (used if routing chooses all, or for iteration)"""
        if not self.has_vectorstore(session_id):
            return {}
            
        session_dir = settings.VECTORS_DIR / session_id
        stores = {}
        
        # We know what files exist from metadata or by listing directory
        for item in session_dir.iterdir():
            if item.is_dir():
                try:
                    safe_name = item.name
                    original_name = safe_name
                    
                    if session_id in self.session_metadata and safe_name in self.session_metadata[session_id].get("documents", {}):
                        original_name = self.session_metadata[session_id]["documents"][safe_name]["original_filename"]
                        
                    stores[original_name] = self.get_vectorstore(session_id, original_name)
                except Exception as e:
                    logger.warning(f"Could not load vectorstore {item.name}: {e}")
                    
        return stores
        
    def get_document_filenames(self, session_id: str) -> List[str]:
        """Get list of original filenames uploaded in this session"""
        if session_id in self.session_metadata and "documents" in self.session_metadata[session_id]:
            return [doc["original_filename"] for doc in self.session_metadata[session_id]["documents"].values()]
            
        # Fallback to directory listing if metadata lost
        session_dir = settings.VECTORS_DIR / session_id
        if session_dir.exists():
            return [d.name for d in session_dir.iterdir() if d.is_dir()]
        return []
    
    def has_vectorstore(self, session_id: str) -> bool:
        """Check if any vectorstore exists for session"""
        session_dir = settings.VECTORS_DIR / session_id
        if session_id in self.vectorstore_cache and self.vectorstore_cache[session_id]:
            return True
        return session_dir.exists() and any(session_dir.iterdir())
    
    def delete_vectorstore(self, session_id: str) -> None:
        """Delete all vectorstores for a session"""
        if session_id in self.vectorstore_cache:
            del self.vectorstore_cache[session_id]
        
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
        
        session_dir = settings.VECTORS_DIR / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"Deleted vectorstores for session {session_id}")
    
    def get_metadata(self, session_id: str) -> dict:
        """Get combined metadata for a session"""
        if session_id not in self.session_metadata:
            return {}
            
        docs = self.session_metadata[session_id].get("documents", {})
        total_pages = sum(d.get("pages", 0) for d in docs.values())
        total_chunks = sum(d.get("chunks", 0) for d in docs.values())
        
        return {
            "files_count": len(docs),
            "pages": total_pages,
            "chunks": total_chunks,
            "documents": docs
        }
    
    def get_active_sessions(self) -> int:
        return len(self.vectorstore_cache)


# Global instance
vector_service = VectorStoreService()
