"""
Minimal FastAPI server for testing RAG functionality
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import logging

# Local imports
from config import settings
from models import Base, User, Bot
from schemas import UserResponse, BotResponse, ChatMessage, ChatResponse
from rag_simple import get_bot_rag_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application...")
    # Create upload directory
    Path("uploads").mkdir(exist_ok=True)
    yield
    logger.info("Shutting down application...")

# FastAPI app
app = FastAPI(
    title="RAG API",
    description="Simple RAG API for testing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG API is running"}

# Simple test bot endpoint
@app.get("/api/v1/test-bot")
async def get_test_bot(db: Session = Depends(get_db)):
    """Get or create a test bot"""
    # Check if test bot exists
    test_bot = db.query(Bot).filter(Bot.name == "Test Bot").first()
    if not test_bot:
        # Create test bot
        test_bot = Bot(
            name="Test Bot",
            description="A test bot for RAG testing",
            user_id=1,  # Assume user 1 exists or create one
            is_active=True
        )
        db.add(test_bot)
        db.commit()
        db.refresh(test_bot)
    
    return {"id": test_bot.id, "name": test_bot.name, "description": test_bot.description}

# File upload endpoint
@app.post("/api/v1/test-upload")
async def upload_test_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a file for testing"""
    try:
        # Get or create test bot
        test_bot = db.query(Bot).filter(Bot.name == "Test Bot").first()
        if not test_bot:
            test_bot = Bot(
                name="Test Bot",
                description="A test bot for RAG testing",
                user_id=1,
                is_active=True
            )
            db.add(test_bot)
            db.commit()
            db.refresh(test_bot)
        
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Add to RAG engine
        rag_engine = get_bot_rag_engine(str(test_bot.id))
        success = rag_engine.add_document(text_content, {"filename": file.filename})
        
        if success:
            return {
                "message": "File uploaded successfully",
                "filename": file.filename,
                "bot_id": test_bot.id,
                "status": "processed"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add document to RAG engine"
            )
            
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a text file"
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

# Chat endpoint
@app.post("/api/v1/test-chat")
async def test_chat(
    chat_request: ChatMessage,
    db: Session = Depends(get_db)
):
    """Chat with the test bot"""
    try:
        # Get test bot
        test_bot = db.query(Bot).filter(Bot.name == "Test Bot").first()
        if not test_bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Test bot not found"
            )
        
        # Get RAG engine
        rag_engine = get_bot_rag_engine(str(test_bot.id))
        
        # Generate response
        response = rag_engine.generate_response(chat_request.message)
        
        return {
            "message": response,
            "bot_id": test_bot.id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
