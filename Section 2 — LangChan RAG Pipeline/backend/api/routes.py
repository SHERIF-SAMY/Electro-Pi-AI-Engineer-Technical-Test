"""
API routes for the RAG chatbot
"""
from typing import List
from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import logging

from ..models.schemas import (
    ChatRequest, ChatResponse, UploadResponse, 
    SessionInfo, HealthResponse
)
from ..services.vector_service import vector_service
from ..services.chat_service import chat_service
from ..utils.helpers import (
    validate_session_id, validate_pdf_upload, 
    save_upload_file, cleanup_temp_file
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the API and dependencies are working
    """
    try:
        # Test Ollama connection
        test_embed = vector_service.embeddings.embed_query("test")
        
        return HealthResponse(
            status="healthy",
            ollama_status="connected",
            embedding_model=settings.EMBEDDING_MODEL,
            llm_model=settings.LLM_MODEL,
            active_sessions=vector_service.get_active_sessions()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "note": f"Make sure Ollama is running with 'ollama pull {settings.EMBEDDING_MODEL}'"
            }
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    files: List[UploadFile] = File(..., description="PDF files to upload (Max 3)"),
    session_id: str = Form(..., description="Session identifier")
):
    """
    Upload and process up to 3 PDF files for RAG
    """
    logger.info(f"📄 Received PDF upload request for session: {session_id} with {len(files)} files")
    
    # Validate session ID
    if not validate_session_id(session_id):
        raise HTTPException(
            status_code=400, 
            detail="Invalid session ID format. Must start with 'session_' or 'portfolio_'"
        )
    
    if len(files) > 3:
        raise HTTPException(status_code=400, detail="Maximum 3 files allowed")
        
    total_pages = 0
    total_chunks = 0
    filenames = []
    
    try:
        for file in files:
            # Validate PDF file
            await validate_pdf_upload(file, settings.MAX_FILE_SIZE)
            
            # Save uploaded PDF temporarily
            file_location = settings.TEMP_UPLOAD_DIR / file.filename
            
            # Save file
            await save_upload_file(file, file_location)
            logger.info(f"💾 Saved PDF temporarily: {file_location}")
            
            # Process PDF and create/update vectorstore
            num_pages, num_chunks = await vector_service.process_pdf(
                str(file_location), 
                session_id,
                file.filename
            )
            
            total_pages += num_pages
            total_chunks += num_chunks
            filenames.append(file.filename)
            
            # Cleanup temporary file immediately after processing
            cleanup_temp_file(str(file_location))
            
        logger.info(f"✅ Successfully processed {len(files)} PDFs for session {session_id}")
        
        return UploadResponse(
            message=f"Successfully uploaded and processed {len(files)} files",
            filenames=filenames,
            files_count=len(filenames),
            pages=total_pages,
            chunks=total_chunks,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing PDFs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG and conversation history
    """
    logger.info(f"💬 Received chat request for session: {request.session_id}")
    logger.info(f"📝 Prompt: {request.prompt[:100]}...")
    
    # Check if vectorstore exists
    if not vector_service.has_vectorstore(request.session_id):
        raise HTTPException(
            status_code=400,
            detail="No PDF has been uploaded for this session. Please upload a PDF first."
        )
    
    try:
        # Process chat — returns (answer, citations, no_context_found)
        answer, sources, no_context_found = await chat_service.chat(
            request.prompt, request.session_id
        )
        
        logger.info(f"✅ Response generated for session {request.session_id}")
        
        return ChatResponse(
            answer=answer,
            sources=sources if sources else None,
            no_context_found=no_context_found,
            session_id=request.session_id,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/info", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """
    Get information about a session
    """
    if not vector_service.has_vectorstore(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionInfo(
        session_id=session_id,
        pdf_info=vector_service.get_metadata(session_id),
        message_count=chat_service.get_message_count(session_id),
        has_vectorstore=vector_service.has_vectorstore(session_id)
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its data
    """
    try:
        # Delete from both services
        chat_service.delete_session(session_id)
        vector_service.delete_vectorstore(session_id)
        
        logger.info(f"✅ Deleted session {session_id}")
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error deleting session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.post("/sessions/{session_id}/clear-history")
async def clear_session_history(session_id: str):
    """
    Clear chat history for a session (keeps vectorstore)
    """
    if not chat_service.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_service.clear_history(session_id)
    
    return {"message": "Chat history cleared successfully"}


@router.get("/sessions")
async def list_sessions():
    """
    List all active sessions
    """
    import os
    sessions = []
    
    if settings.VECTORS_DIR.exists():
        for session_dir in settings.VECTORS_DIR.iterdir():
            if session_dir.is_dir():
                sessions.append({
                    "session_id": session_dir.name,
                    "has_vectorstore": True,
                    "message_count": chat_service.get_message_count(session_dir.name)
                })
    
    return {"sessions": sessions, "count": len(sessions)}
