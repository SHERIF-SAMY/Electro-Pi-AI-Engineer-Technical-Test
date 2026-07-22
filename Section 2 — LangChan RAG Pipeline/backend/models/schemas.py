"""
Pydantic models for request and response validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    prompt: str = Field(..., min_length=1, max_length=5000, description="User's question")
    session_id: str = Field(..., min_length=5, description="Unique session identifier")
    
    @field_validator('prompt')
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Prompt cannot be empty or whitespace')
        return v.strip()
    
    @field_validator('session_id')
    @classmethod
    def validate_session_format(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 5:
            raise ValueError('Session ID must be at least 5 characters')
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', stripped):
            raise ValueError('Session ID may only contain letters, digits, underscores, and hyphens')
        return stripped


class Citation(BaseModel):
    """Structured citation pointing to a specific retrieved chunk"""
    chunk_id: str = Field(..., description="Unique chunk identifier: {session_id}_p{page}_c{index}")
    filename: str = Field(..., description="Source PDF filename")
    page: int = Field(..., description="Page number (0-indexed) within the PDF")
    chunk_index: int = Field(..., description="Position of this chunk among all chunks")
    total_chunks: int = Field(..., description="Total number of chunks in the document")
    content_preview: str = Field(..., description="First 200 characters of the chunk text")
    score: Optional[float] = Field(default=None, description="Cosine similarity score (higher = more relevant)")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(..., description="AI generated answer")
    sources: Optional[List[Citation]] = Field(default=None, description="Structured citations for retrieved chunks")
    no_context_found: bool = Field(default=False, description="True when no relevant chunks were found above the similarity threshold")
    session_id: str = Field(..., description="Session identifier")


class UploadResponse(BaseModel):
    """Response model for PDF upload endpoint"""
    message: str
    filenames: List[str]
    files_count: int
    pages: int
    chunks: int
    session_id: str


class SessionInfo(BaseModel):
    """Session information response"""
    session_id: str
    pdf_info: Optional[dict] = None
    message_count: int
    has_vectorstore: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ollama_status: str
    embedding_model: str
    llm_model: str
    active_sessions: int


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
    error_type: Optional[str] = None
