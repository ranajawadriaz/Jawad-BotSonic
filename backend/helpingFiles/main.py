from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from rag import RAGChatbot


# Initialize FastAPI application
app = FastAPI(title="RAG Chatbot API", description="A RAG-based chatbot for document Q&A")

# Initialize Jinja2 templates for HTML rendering
templates = Jinja2Templates(directory="templates")

# Initialize RAG chatbot instance
rag_bot = RAGChatbot()


# Pydantic model for chat messages
class ChatMessage(BaseModel):
    message: str

# Pydantic model for URL input
class URLInput(BaseModel):
    url: str


# Root endpoint - serves the chatbot HTML interface
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chatbot interface"""
    return templates.TemplateResponse("bot.html", {"request": request})


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running"""
    return {"status": "healthy", "message": "RAG Chatbot API is running"}


# File upload endpoint for document processing
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process documents (PDF, Excel, TXT) to store in vector database
    """
    try:
        # Check if file type is supported
        allowed_extensions = ['.pdf', '.xlsx', '.xls', '.txt']
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PDF, Excel, or TXT files."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process and store document in ChromaDB
        result = rag_bot.process_and_store_document(file_content, file.filename)
        
        # Check if processing was successful
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content={
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "details": result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# Chat endpoint for user queries
@app.post("/chat")
async def chat_with_bot(chat_message: ChatMessage):
    """
    Send a message to the RAG chatbot and get AI-generated response
    """
    try:
        # Validate input message
        if not chat_message.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Generate response using RAG system
        response = rag_bot.generate_response(chat_message.message)
        
        return JSONResponse(content={
            "user_message": chat_message.message,
            "bot_response": response,
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# Get database information endpoint
@app.get("/db-info")
async def get_database_info():
    """
    Get information about the current state of the vector database
    """
    try:
        # Get collection information from ChromaDB
        info = rag_bot.get_collection_info()
        
        return JSONResponse(content={
            "database_info": info,
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database info: {str(e)}")


# Clear database endpoint (optional - for testing/development)
@app.delete("/clear-db")
async def clear_database():
    """
    Clear all documents from the vector database (use with caution)
    """
    try:
        # Delete and recreate collection to clear all data
        rag_bot.chroma_client.delete_collection(name="documents")
        rag_bot.collection = rag_bot.chroma_client.create_collection(
            name="documents"
        )
        
        return JSONResponse(content={
            "message": "Database cleared successfully",
            "status": "success"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")


# URL processing endpoint for website scraping
@app.post("/upload-url")
async def upload_url(url_input: URLInput):
    """
    Process a website URL by scraping its content and storing in vector database
    """
    try:
        # Validate URL format
        if not url_input.url.strip():
            raise HTTPException(status_code=400, detail="URL cannot be empty")
        
        # Ensure URL has proper protocol
        url = url_input.url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Process URL and store in ChromaDB
        result = rag_bot.process_url(url)
        
        # Check if processing was successful
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content={
            "message": "Website content scraped and processed successfully",
            "url": url,
            "details": result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL: {str(e)}")


# Run the application if this file is executed directly
if __name__ == "__main__":
    # Start FastAPI server with uvicorn
    uvicorn.run(
        "main:app",      # Use import string format
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,       # Port number
        reload=True      # Auto-reload on code changes (for development)
    )