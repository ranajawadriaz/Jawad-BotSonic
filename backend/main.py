from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Cookie, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import asynccontextmanager
import uvicorn

from config import settings, get_database_url
from models import Base, User, Bot, Conversation, Message
from schemas import (
    UserCreate, UserLogin, UserUpdate, UserResponse,
    DemoRequestCreate, DemoRequestResponse,
    BotCreate, BotUpdate, BotResponse,
    GuidelineCreate, GuidelineUpdate, GuidelineResponse,
    StarterQuestionCreate, StarterQuestionUpdate, StarterQuestionResponse,
    SourceResponse, ChatMessage, ChatResponse,
    ConversationResponse, MessageResponse,
    URLInput, FileUploadResponse,
    BotAnalytics, UserAnalytics,
    Token, APIResponse, ErrorResponse,
    EmbeddingConfig
)
from jwt_ import (
    require_authentication, require_authentication_cookie, require_authentication_flexible, create_token_pair,
    get_current_user_from_cookie, get_current_user, get_current_user_flexible, TokenData
)
from controllers import (
    AuthController, DemoController, BotController,
    GuidelineController, SourceController, ChatController,
    AnalyticsController
)

# Database setup
engine = create_engine(get_database_url())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# Security
security = HTTPBearer()

def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("🚀 Botsonic Clone API starting up...")
    print(f"📊 Database: {get_database_url()}")
    print(f"🔧 Environment: {'Development' if settings.DEBUG else 'Production'}")
    
    yield
    
    # Shutdown
    print("🛑 Botsonic Clone API shutting down...")

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="A Botsonic clone with RAG chatbot functionality",
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Add CORS middleware - this needs to be before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Additional CORS middleware for embedding (allows all origins without credentials)
@app.middleware("http")
async def embedding_cors_middleware(request: Request, call_next):
    # For embedding endpoints, we need to allow all origins
    if request.url.path.startswith("/api/v1/embed/") or request.url.path.startswith("/api/v1/chat/"):
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        # Process the request
        response = await call_next(request)
        
        # Add CORS headers to the response
        response.headers["Access-Control-Allow-Origin"] = origin or "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        return response
    
    # For non-embedding endpoints, just proceed normally (CORSMiddleware will handle it)
    return await call_next(request)

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    return """
    <html>
        <head>
            <title>Botsonic Clone API</title>
        </head>
        <body>
            <h1>🤖 Botsonic Clone API</h1>
            <p>Welcome to the Botsonic Clone API!</p>
            <ul>
                <li><a href="/docs">📚 API Documentation (Swagger)</a></li>
                <li><a href="/redoc">📖 ReDoc Documentation</a></li>
                <li><a href="/health">🔍 Health Check</a></li>
            </ul>
        </body>
    </html>
    """

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "debug": settings.DEBUG
    }

# Authentication endpoints
@app.post("/api/v1/auth/register", response_model=dict)
async def register(
    user_data: UserCreate,
    response: Response,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    try:
        result = AuthController.register_user(user_data, db)
        
        # Set HTTP-only cookies for tokens
        response.set_cookie(
            key="access_token",
            value=result["tokens"]["access_token"],
            max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            httponly=True,
            secure=not settings.DEBUG,
            samesite="lax"
        )
        response.set_cookie(
            key="refresh_token",
            value=result["tokens"]["refresh_token"],
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
            httponly=True,
            secure=not settings.DEBUG,
            samesite="lax"
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@app.post("/api/v1/auth/login", response_model=dict)
async def login(
    login_data: UserLogin,
    response: Response,
    db: Session = Depends(get_db)
):
    """Login user"""
    try:
        result = AuthController.login_user(login_data, db)
        
        # Set HTTP-only cookies for tokens
        cookie_max_age = 30 * 24 * 60 * 60 if login_data.remember_me else settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        
        response.set_cookie(
            key="access_token",
            value=result["tokens"]["access_token"],
            max_age=cookie_max_age,
            httponly=True,
            secure=not settings.DEBUG,
            samesite="lax"
        )
        response.set_cookie(
            key="refresh_token",
            value=result["tokens"]["refresh_token"],
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
            httponly=True,
            secure=not settings.DEBUG,
            samesite="lax"
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/api/v1/auth/logout")
async def logout(response: Response):
    """Logout user"""
    response.delete_cookie(key="access_token")
    response.delete_cookie(key="refresh_token")
    return {"message": "Logged out successfully"}

def get_current_user_dependency(
    db: Session = Depends(get_db),
    token_data: TokenData = Depends(get_current_user_flexible)
) -> User:
    """Dependency wrapper for getting current user with flexible auth"""
    return get_current_user(db, token_data)

@app.get("/api/v1/auth/me", response_model=UserResponse)
async def get_current_user_endpoint(
    current_user: User = Depends(get_current_user_dependency)
):
    """Get current user profile"""
    return AuthController.get_current_user_profile(current_user)

@app.put("/api/v1/auth/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    return AuthController.update_user_profile(user_data, current_user, db)

# Demo request endpoints
@app.post("/api/v1/demo", response_model=DemoRequestResponse)
async def submit_demo_request(
    demo_data: DemoRequestCreate,
    db: Session = Depends(get_db),
    current_user = None
):
    """Submit a demo request (can be anonymous or authenticated)"""
    try:
        # Try to get current user from cookie, but don't require it
        access_token = None
        if hasattr(demo_data, 'access_token'):
            access_token = demo_data.access_token
        
        if access_token:
            try:
                current_user = require_authentication_cookie(access_token)
            except:
                current_user = None
        
        return DemoController.submit_demo_request(demo_data, db, current_user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demo request submission failed: {str(e)}"
        )

# Bot management endpoints
@app.post("/api/v1/bots", response_model=BotResponse)
async def create_bot(
    bot_data: BotCreate,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Create a new bot"""
    return BotController.create_bot(bot_data, current_user, db)

@app.get("/api/v1/bots", response_model=list[BotResponse])
async def get_user_bots(
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get all bots for the current user"""
    return BotController.get_user_bots(current_user, db)

@app.get("/api/v1/bots/{bot_id}", response_model=BotResponse)
async def get_bot(
    bot_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get a specific bot"""
    return BotController.get_bot_by_id(bot_id, current_user, db)

@app.put("/api/v1/bots/{bot_id}", response_model=BotResponse)
async def update_bot(
    bot_id: int,
    bot_data: BotUpdate,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Update bot configuration"""
    return BotController.update_bot(bot_id, bot_data, current_user, db)

@app.delete("/api/v1/bots/{bot_id}")
async def delete_bot(
    bot_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Delete a bot"""
    return BotController.delete_bot(bot_id, current_user, db)

@app.get("/api/v1/bots/{bot_id}/embed", response_model=EmbeddingConfig)
async def get_bot_embedding_config(
    bot_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get embedding configuration for a bot"""
    return BotController.get_embedding_config(bot_id, current_user, db)

# Bot guidelines endpoints
@app.post("/api/v1/bots/{bot_id}/guidelines", response_model=GuidelineResponse)
async def create_guideline(
    bot_id: int,
    guideline_data: GuidelineCreate,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Create a new guideline for a bot"""
    return GuidelineController.create_guideline(bot_id, guideline_data, current_user, db)

@app.get("/api/v1/bots/{bot_id}/guidelines", response_model=list[GuidelineResponse])
async def get_bot_guidelines(
    bot_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get all guidelines for a bot"""
    return GuidelineController.get_bot_guidelines(bot_id, current_user, db)

@app.put("/api/v1/guidelines/{guideline_id}", response_model=GuidelineResponse)
async def update_guideline(
    guideline_id: int,
    guideline_data: GuidelineUpdate,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Update a guideline"""
    return GuidelineController.update_guideline(guideline_id, guideline_data, current_user, db)

@app.delete("/api/v1/guidelines/{guideline_id}")
async def delete_guideline(
    guideline_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Delete a guideline"""
    return GuidelineController.delete_guideline(guideline_id, current_user, db)

# Bot sources endpoints
@app.post("/api/v1/bots/{bot_id}/sources/files", response_model=FileUploadResponse)
async def upload_file(
    bot_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Upload a file to a bot"""
    return SourceController.upload_file(bot_id, file, current_user, db)

@app.post("/api/v1/bots/{bot_id}/sources/urls", response_model=SourceResponse)
async def add_url(
    bot_id: int,
    url_data: URLInput,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Add a URL to a bot"""
    return SourceController.add_url(bot_id, url_data, current_user, db)

@app.get("/api/v1/bots/{bot_id}/sources", response_model=list[SourceResponse])
async def get_bot_sources(
    bot_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get all sources for a bot"""
    return SourceController.get_bot_sources(bot_id, current_user, db)

# Chat endpoints
@app.post("/api/v1/bots/{bot_id}/chat", response_model=ChatResponse)
async def chat_with_bot(
    bot_id: int,
    chat_data: ChatMessage,
    token_data: TokenData = Depends(require_authentication_flexible),
    db: Session = Depends(get_db)
):
    """Chat with a bot (authenticated)"""
    current_user = get_current_user(db, token_data)
    return ChatController.chat_with_bot(bot_id, chat_data, current_user, db)

@app.post("/api/v1/chat/{api_token}", response_model=ChatResponse)
async def chat_with_bot_api(
    api_token: str,
    chat_data: ChatMessage,
    db: Session = Depends(get_db)
):
    """Chat with a bot using API token (for embedding)"""
    return ChatController.chat_with_bot_api(api_token, chat_data, db)

@app.get("/api/v1/embed/bot/{api_token}")
async def get_bot_embed_info(
    api_token: str,
    db: Session = Depends(get_db)
):
    """Get basic bot information for embedding (public endpoint)"""
    bot = db.query(Bot).filter(
        Bot.api_token == api_token,
        Bot.is_active == True
    ).first()
    
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found or inactive"
        )
    
    return {
        "id": bot.id,
        "name": bot.name,
        "widget_name": bot.widget_name,
        "api_token": bot.api_token,
        "accent_color": bot.accent_color,
        "widget_position": bot.widget_position,
        "input_placeholder": bot.input_placeholder,
        "user_form_enabled": bot.user_form_enabled,
        "user_form_fields": bot.user_form_fields
    }

# Analytics endpoints
@app.get("/api/v1/bots/{bot_id}/analytics", response_model=BotAnalytics)
async def get_bot_analytics(
    bot_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get analytics for a bot"""
    return AnalyticsController.get_bot_analytics(bot_id, current_user, db)

@app.get("/api/v1/analytics", response_model=UserAnalytics)
async def get_user_analytics(
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get analytics for the current user"""
    return AnalyticsController.get_user_analytics(current_user, db)

# Conversation endpoints
@app.get("/api/v1/bots/{bot_id}/conversations", response_model=list[ConversationResponse])
async def get_bot_conversations(
    bot_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get all conversations for a bot"""
    # Verify bot ownership
    bot = db.query(Bot).filter(
        Bot.id == bot_id,
        Bot.owner_id == current_user.id,
        Bot.is_active == True
    ).first()
    
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bot not found"
        )
    
    conversations = db.query(Conversation).filter(
        Conversation.bot_id == bot_id
    ).order_by(Conversation.started_at.desc()).all()
    
    return [ConversationResponse.from_orm(conv) for conv in conversations]

@app.get("/api/v1/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_conversation_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user_dependency),
    db: Session = Depends(get_db)
):
    """Get all messages for a conversation"""
    # Verify conversation belongs to user's bot
    conversation = db.query(Conversation).join(Bot).filter(
        Conversation.id == conversation_id,
        Bot.owner_id == current_user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.asc()).all()
    
    return [MessageResponse.from_orm(msg) for msg in messages]

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "path": str(request.url),
            "details": str(exc) if settings.DEBUG else None
        }
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )
