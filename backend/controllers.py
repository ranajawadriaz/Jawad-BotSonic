from fastapi import HTTPException, UploadFile, File, Depends, Form, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func
from typing import List, Optional
import uuid
from datetime import datetime, timedelta

from models import (
    User, DemoRequest, Bot, BotGuideline, StarterQuestion, 
    BotSource, Conversation, Message, Usage,
    generate_api_token, generate_collection_name
)
from schemas import (
    UserCreate, UserUpdate, UserResponse, UserLogin,
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
    authenticate_user, create_token_pair, get_password_hash,
    verify_password, require_authentication, require_authentication_cookie
)
from rag import get_bot_rag_engine
from config import settings


class AuthController:
    """Authentication and user management controllers"""
    
    @staticmethod
    def register_user(user_data: UserCreate, db: Session) -> dict:
        """Register a new user"""
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        db_user = User(
            full_name=user_data.full_name,
            email=user_data.email,
            hashed_password=hashed_password,
            company_name=user_data.company_name,
            job_title=user_data.job_title,
            country=user_data.country,
            monthly_queries=user_data.monthly_queries,
            budget=user_data.budget
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        # Create tokens
        tokens = create_token_pair(db_user)
        
        return {
            "message": "User registered successfully",
            "user": UserResponse.from_orm(db_user),
            "tokens": tokens
        }
    
    @staticmethod
    def login_user(login_data: UserLogin, db: Session) -> dict:
        """Authenticate user and return tokens"""
        user = authenticate_user(db, login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user account"
            )
        
        tokens = create_token_pair(user)
        
        return {
            "message": "Login successful",
            "user": UserResponse.from_orm(user),
            "tokens": tokens
        }
    
    @staticmethod
    def get_current_user_profile(current_user: User) -> UserResponse:
        """Get current user profile"""
        return UserResponse.from_orm(current_user)
    
    @staticmethod
    def update_user_profile(user_data: UserUpdate, current_user: User, db: Session) -> UserResponse:
        """Update user profile"""
        for field, value in user_data.dict(exclude_unset=True).items():
            setattr(current_user, field, value)
        
        current_user.updated_at = func.now()
        db.commit()
        db.refresh(current_user)
        
        return UserResponse.from_orm(current_user)


class DemoController:
    """Demo request management controllers"""
    
    @staticmethod
    def submit_demo_request(demo_data: DemoRequestCreate, db: Session, current_user: User = None) -> DemoRequestResponse:
        """Submit a demo request"""
        db_demo = DemoRequest(
            first_name=demo_data.first_name,
            last_name=demo_data.last_name,
            work_email=demo_data.work_email,
            company_name=demo_data.company_name,
            job_title=demo_data.job_title,
            country=demo_data.country,
            monthly_queries=demo_data.monthly_queries,
            budget=demo_data.budget,
            additional_notes=demo_data.additional_notes,
            user_id=current_user.id if current_user else None
        )
        
        db.add(db_demo)
        db.commit()
        db.refresh(db_demo)
        
        return DemoRequestResponse.from_orm(db_demo)


class BotController:
    """Bot management controllers"""
    
    @staticmethod
    def create_bot(bot_data: BotCreate, current_user: User, db: Session) -> BotResponse:
        """Create a new bot for the user"""
        # Generate unique collection name and API token
        collection_name = generate_collection_name(bot_data.name, current_user.id)
        api_token = generate_api_token()
        
        db_bot = Bot(
            name=bot_data.name,
            description=bot_data.description,
            collection_name=collection_name,
            api_token=api_token,
            owner_id=current_user.id
        )
        
        db.add(db_bot)
        db.commit()
        db.refresh(db_bot)
        
        return BotResponse.from_orm(db_bot)
    
    @staticmethod
    def get_user_bots(current_user: User, db: Session) -> List[BotResponse]:
        """Get all bots for the current user"""
        bots = db.query(Bot).filter(Bot.owner_id == current_user.id, Bot.is_active == True).all()
        return [BotResponse.from_orm(bot) for bot in bots]
    
    @staticmethod
    def get_bot_by_id(bot_id: int, current_user: User, db: Session) -> BotResponse:
        """Get a specific bot by ID"""
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
        
        return BotResponse.from_orm(bot)
    
    @staticmethod
    def update_bot(bot_id: int, bot_data: BotUpdate, current_user: User, db: Session) -> BotResponse:
        """Update bot configuration"""
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
        
        for field, value in bot_data.dict(exclude_unset=True).items():
            setattr(bot, field, value)
        
        bot.updated_at = func.now()
        db.commit()
        db.refresh(bot)
        
        return BotResponse.from_orm(bot)
    
    @staticmethod
    def delete_bot(bot_id: int, current_user: User, db: Session) -> dict:
        """Delete a bot (soft delete)"""
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
        
        bot.is_active = False
        bot.updated_at = func.now()
        db.commit()
        
        return {"message": "Bot deleted successfully"}
    
    @staticmethod
    def get_embedding_config(bot_id: int, current_user: User, db: Session) -> EmbeddingConfig:
        """Get embedding configuration for a bot"""
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
        
        base_url = "http://localhost:8000"  # Should be configurable
        
        return EmbeddingConfig(
            bot_id=bot.id,
            collection_name=bot.collection_name,
            iframe_url=f"{base_url}/embed/iframe/{bot.api_token}",
            javascript_snippet=f'<script src="{base_url}/embed/widget.js" data-bot-token="{bot.api_token}"></script>',
            api_endpoint=f"{base_url}/api/v1/chat/{bot.api_token}",
            api_token=bot.api_token
        )


class GuidelineController:
    """Bot guidelines management controllers"""
    
    @staticmethod
    def create_guideline(bot_id: int, guideline_data: GuidelineCreate, current_user: User, db: Session) -> GuidelineResponse:
        """Create a new guideline for a bot"""
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
        
        db_guideline = BotGuideline(
            title=guideline_data.title,
            content=guideline_data.content,
            bot_id=bot_id
        )
        
        db.add(db_guideline)
        db.commit()
        db.refresh(db_guideline)
        
        return GuidelineResponse.from_orm(db_guideline)
    
    @staticmethod
    def get_bot_guidelines(bot_id: int, current_user: User, db: Session) -> List[GuidelineResponse]:
        """Get all guidelines for a bot"""
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
        
        guidelines = db.query(BotGuideline).filter(BotGuideline.bot_id == bot_id).all()
        return [GuidelineResponse.from_orm(guideline) for guideline in guidelines]
    
    @staticmethod
    def update_guideline(guideline_id: int, guideline_data: GuidelineUpdate, current_user: User, db: Session) -> GuidelineResponse:
        """Update a guideline"""
        # Get guideline and verify ownership through bot
        guideline = db.query(BotGuideline).join(Bot).filter(
            BotGuideline.id == guideline_id,
            Bot.owner_id == current_user.id,
            Bot.is_active == True
        ).first()
        
        if not guideline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline not found"
            )
        
        for field, value in guideline_data.dict(exclude_unset=True).items():
            setattr(guideline, field, value)
        
        db.commit()
        db.refresh(guideline)
        
        return GuidelineResponse.from_orm(guideline)
    
    @staticmethod
    def delete_guideline(guideline_id: int, current_user: User, db: Session) -> dict:
        """Delete a guideline"""
        guideline = db.query(BotGuideline).join(Bot).filter(
            BotGuideline.id == guideline_id,
            Bot.owner_id == current_user.id,
            Bot.is_active == True
        ).first()
        
        if not guideline:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guideline not found"
            )
        
        db.delete(guideline)
        db.commit()
        
        return {"message": "Guideline deleted successfully"}


class SourceController:
    """Bot sources (files/URLs) management controllers"""
    
    @staticmethod
    def upload_file(bot_id: int, file: UploadFile, current_user: User, db: Session) -> FileUploadResponse:
        """Upload and process a file for a bot"""
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
        
        # Check file size
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Check file extension
        file_extension = '.' + file.filename.split('.')[-1].lower()
        if file_extension not in settings.ALLOWED_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_FILE_EXTENSIONS)}"
            )
        
        # Create source record
        db_source = BotSource(
            source_type="file",
            source_name=file.filename,
            file_size=file.size,
            processing_status="processing",
            bot_id=bot_id
        )
        
        db.add(db_source)
        db.commit()
        db.refresh(db_source)
        
        # Process file with RAG engine
        try:
            file_content = file.file.read()
            rag_engine = get_bot_rag_engine(bot)
            result = rag_engine.process_and_store_document(
                file_content=file_content,
                filename=file.filename,
                db=db
            )
            
            if "error" in result:
                db_source.processing_status = "failed"
                db_source.error_message = result["error"]
                db.commit()
                
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result["error"]
                )
            
            return FileUploadResponse(
                message=result["message"],
                filename=file.filename,
                file_size=file.size,
                source_id=db_source.id,
                processing_status="completed"
            )
            
        except Exception as e:
            db_source.processing_status = "failed"
            db_source.error_message = str(e)
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing file: {str(e)}"
            )
    
    @staticmethod
    def add_url(bot_id: int, url_data: URLInput, current_user: User, db: Session) -> SourceResponse:
        """Add and process a URL for a bot"""
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
        
        # Ensure URL has protocol
        url = url_data.url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Create source record
        db_source = BotSource(
            source_type="url",
            source_name=url,
            source_url=url,
            processing_status="processing",
            bot_id=bot_id
        )
        
        db.add(db_source)
        db.commit()
        db.refresh(db_source)
        
        # Process URL with RAG engine
        try:
            rag_engine = get_bot_rag_engine(bot)
            result = rag_engine.process_and_store_document(url=url, db=db)
            
            if "error" in result:
                db_source.processing_status = "failed"
                db_source.error_message = result["error"]
                db.commit()
                
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result["error"]
                )
            
            return SourceResponse.from_orm(db_source)
            
        except Exception as e:
            db_source.processing_status = "failed"
            db_source.error_message = str(e)
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing URL: {str(e)}"
            )
    
    @staticmethod
    def get_bot_sources(bot_id: int, current_user: User, db: Session) -> List[SourceResponse]:
        """Get all sources for a bot"""
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
        
        sources = db.query(BotSource).filter(BotSource.bot_id == bot_id).all()
        return [SourceResponse.from_orm(source) for source in sources]


class ChatController:
    """Chat functionality controllers"""
    
    @staticmethod
    def chat_with_bot(bot_id: int, chat_data: ChatMessage, current_user: User, db: Session) -> ChatResponse:
        """Chat with a bot (authenticated)"""
        # Verify bot ownership and load guidelines
        bot = db.query(Bot).options(joinedload(Bot.guidelines)).filter(
            Bot.id == bot_id,
            Bot.owner_id == current_user.id,
            Bot.is_active == True
        ).first()
        
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bot not found"
            )
        
        return ChatController._process_chat_message(bot, chat_data, db, current_user.email)
    
    @staticmethod
    def chat_with_bot_api(api_token: str, chat_data: ChatMessage, db: Session) -> ChatResponse:
        """Chat with a bot using API token (for embedding)"""
        # Get bot by API token and load guidelines
        bot = db.query(Bot).options(joinedload(Bot.guidelines)).filter(
            Bot.api_token == api_token,
            Bot.is_active == True
        ).first()
        
        if not bot:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API token"
            )
        
        return ChatController._process_chat_message(bot, chat_data, db, chat_data.user_identifier)
    
    @staticmethod
    def _process_chat_message(bot: Bot, chat_data: ChatMessage, db: Session, user_identifier: str = None) -> ChatResponse:
        """Internal method to process chat messages"""
        # Get or create conversation
        session_id = chat_data.session_id or str(uuid.uuid4())
        
        conversation = db.query(Conversation).filter(
            Conversation.session_id == session_id,
            Conversation.bot_id == bot.id
        ).first()
        
        if not conversation:
            conversation = Conversation(
                session_id=session_id,
                bot_id=bot.id,
                user_identifier=user_identifier,
                user_form_data=chat_data.user_form_data
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
        elif chat_data.user_form_data and not conversation.user_form_data:
            # Update conversation with user form data if not already set
            conversation.user_form_data = chat_data.user_form_data
            db.commit()
        
        # Save user message
        user_message = Message(
            content=chat_data.message,
            message_type="user",
            conversation_id=conversation.id
        )
        db.add(user_message)
        
        # Generate bot response using RAG
        rag_engine = get_bot_rag_engine(bot)
        response_data = rag_engine.generate_response(chat_data.message, session_id)
        
        # Save bot message
        bot_message = Message(
            content=response_data["response"],
            message_type="bot",
            sources_used=response_data.get("sources_used"),
            response_time_ms=response_data.get("response_time_ms"),
            conversation_id=conversation.id
        )
        db.add(bot_message)
        
        # Update bot statistics
        bot.total_messages += 2  # user + bot message
        if conversation.status == "active" and db.query(Message).filter(Message.conversation_id == conversation.id).count() == 2:
            bot.total_conversations += 1
        
        db.commit()
        
        return ChatResponse(
            bot_response=response_data["response"],
            session_id=session_id,
            sources_used=response_data.get("sources_used"),
            response_time_ms=response_data.get("response_time_ms")
        )


class AnalyticsController:
    """Analytics and reporting controllers"""
    
    @staticmethod
    def get_bot_analytics(bot_id: int, current_user: User, db: Session) -> BotAnalytics:
        """Get analytics for a specific bot"""
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
        
        # Calculate time-based metrics
        now = datetime.utcnow()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # Messages today
        messages_today = db.query(Message).join(Conversation).filter(
            Conversation.bot_id == bot_id,
            Message.created_at >= today
        ).count()
        
        # Messages this week
        messages_this_week = db.query(Message).join(Conversation).filter(
            Conversation.bot_id == bot_id,
            Message.created_at >= week_ago
        ).count()
        
        # Messages this month
        messages_this_month = db.query(Message).join(Conversation).filter(
            Conversation.bot_id == bot_id,
            Message.created_at >= month_ago
        ).count()
        
        # Average response time
        avg_response_time = db.query(func.avg(Message.response_time_ms)).join(Conversation).filter(
            Conversation.bot_id == bot_id,
            Message.message_type == "bot",
            Message.response_time_ms.isnot(None)
        ).scalar()
        
        return BotAnalytics(
            total_conversations=bot.total_conversations,
            total_messages=bot.total_messages,
            messages_today=messages_today,
            messages_this_week=messages_this_week,
            messages_this_month=messages_this_month,
            average_response_time_ms=avg_response_time,
            top_sources_used=[],  # TODO: Implement source usage tracking
            conversation_trends=[]  # TODO: Implement trend analysis
        )
    
    @staticmethod
    def get_user_analytics(current_user: User, db: Session) -> UserAnalytics:
        """Get analytics for the current user"""
        # Count user's bots
        total_bots = db.query(Bot).filter(Bot.owner_id == current_user.id, Bot.is_active == True).count()
        
        # Total conversations across all bots
        total_conversations = db.query(func.sum(Bot.total_conversations)).filter(
            Bot.owner_id == current_user.id,
            Bot.is_active == True
        ).scalar() or 0
        
        # Total messages across all bots
        total_messages = db.query(func.sum(Bot.total_messages)).filter(
            Bot.owner_id == current_user.id,
            Bot.is_active == True
        ).scalar() or 0
        
        # Usage this month
        now = datetime.utcnow()
        month_ago = now - timedelta(days=30)
        
        usage_this_month = db.query(Message).join(Conversation).join(Bot).filter(
            Bot.owner_id == current_user.id,
            Message.created_at >= month_ago
        ).count()
        
        # Plan limits (simplified)
        plan_limits = {
            "starter": {"bots": 1, "messages_per_month": 1000},
            "professional": {"bots": 5, "messages_per_month": 10000},
            "advanced": {"bots": 20, "messages_per_month": 50000},
            "enterprise": {"bots": -1, "messages_per_month": -1}  # Unlimited
        }
        
        return UserAnalytics(
            total_bots=total_bots,
            total_conversations=total_conversations,
            total_messages=total_messages,
            subscription_plan=current_user.subscription_plan,
            usage_this_month=usage_this_month,
            plan_limits=plan_limits.get(current_user.subscription_plan, plan_limits["starter"])
        )
