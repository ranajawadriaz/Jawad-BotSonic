from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()


class User(Base):
    """User model for client accounts"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    company_name = Column(String(255), nullable=True)
    job_title = Column(String(255), nullable=True)
    country = Column(String(100), nullable=True)
    monthly_queries = Column(String(100), nullable=True)
    budget = Column(String(100), nullable=True)
    subscription_plan = Column(String(50), default="starter")  # starter, professional, advanced, enterprise
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    bots = relationship("Bot", back_populates="owner", cascade="all, delete-orphan")
    demo_requests = relationship("DemoRequest", back_populates="user")


class DemoRequest(Base):
    """Demo request model for book-a-demo functionality"""
    __tablename__ = "demo_requests"
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(255), nullable=False)
    last_name = Column(String(255), nullable=False)
    work_email = Column(String(255), nullable=False)
    company_name = Column(String(255), nullable=False)
    job_title = Column(String(255), nullable=False)
    country = Column(String(100), nullable=False)
    monthly_queries = Column(String(100), nullable=False)
    budget = Column(String(100), nullable=False)
    additional_notes = Column(Text, nullable=True)
    status = Column(String(50), default="pending")  # pending, scheduled, completed, cancelled
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign key (optional - if user is logged in)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="demo_requests")


class Bot(Base):
    """Bot model for individual chatbots created by users"""
    __tablename__ = "bots"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    collection_name = Column(String(255), unique=True, nullable=False)  # ChromaDB collection name
    api_token = Column(String(255), unique=True, nullable=False)  # For REST API access
    
    # Bot Configuration
    widget_name = Column(String(255), default="AI Assistant")
    accent_color = Column(String(7), default="#007bff")  # Hex color
    widget_icon = Column(String(255), nullable=True)  # Icon URL or name
    widget_position = Column(String(20), default="bottom-right")  # bottom-right, bottom-left, etc.
    input_placeholder = Column(String(255), default="Type your message...")
    
    # Response Configuration
    response_length = Column(String(20), default="medium")  # short, medium, long
    show_sources = Column(Boolean, default=True)
    
    # Rate Limiting
    rate_limit_per_hour = Column(Integer, default=100)
    
    # Working Hours Configuration
    working_hours_enabled = Column(Boolean, default=False)
    working_hours_config = Column(JSON, nullable=True)  # Store working hours as JSON
    
    # Form Configuration
    user_form_enabled = Column(Boolean, default=False)
    user_form_fields = Column(JSON, nullable=True)  # Store form fields as JSON
    
    # Email Configuration
    email_notifications_enabled = Column(Boolean, default=False)
    support_email = Column(String(255), nullable=True)
    transcript_email = Column(String(255), nullable=True)
    
    # Analytics
    total_conversations = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign key
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    owner = relationship("User", back_populates="bots")
    conversations = relationship("Conversation", back_populates="bot", cascade="all, delete-orphan")
    guidelines = relationship("BotGuideline", back_populates="bot", cascade="all, delete-orphan")
    starter_questions = relationship("StarterQuestion", back_populates="bot", cascade="all, delete-orphan")
    sources = relationship("BotSource", back_populates="bot", cascade="all, delete-orphan")


class BotGuideline(Base):
    """Guidelines for bot behavior"""
    __tablename__ = "bot_guidelines"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign key
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    bot = relationship("Bot", back_populates="guidelines")


class StarterQuestion(Base):
    """Starter questions for bots"""
    __tablename__ = "starter_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(String(500), nullable=False)
    order_index = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign key
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    bot = relationship("Bot", back_populates="starter_questions")


class BotSource(Base):
    """Sources (files/URLs) added to bots"""
    __tablename__ = "bot_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(20), nullable=False)  # file, url
    source_name = Column(String(255), nullable=False)  # filename or URL
    source_url = Column(Text, nullable=True)  # For URL sources
    file_size = Column(Integer, nullable=True)  # For file sources
    processing_status = Column(String(20), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)  # Number of chunks created
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Foreign key
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    bot = relationship("Bot", back_populates="sources")


class Conversation(Base):
    """Conversation sessions with bots"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False)
    user_identifier = Column(String(255), nullable=True)  # Email, name, or anonymous ID
    user_form_data = Column(JSON, nullable=True)  # User form submission data
    status = Column(String(20), default="active")  # active, closed
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Foreign key
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    
    # Relationships
    bot = relationship("Bot", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """Individual messages in conversations"""
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    message_type = Column(String(20), nullable=False)  # user, bot
    sources_used = Column(JSON, nullable=True)  # Sources referenced in bot response
    response_time_ms = Column(Integer, nullable=True)  # Response time for bot messages
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign key
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    conversation = relationship("Conversation", back_populates="messages")


class Usage(Base):
    """Usage tracking for analytics"""
    __tablename__ = "usage"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    bot_id = Column(Integer, ForeignKey("bots.id"), nullable=False)
    action_type = Column(String(50), nullable=False)  # message_sent, file_uploaded, url_processed, etc.
    action_metadata = Column(JSON, nullable=True)  # Additional tracking data
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")
    bot = relationship("Bot")


def generate_api_token() -> str:
    """Generate a unique API token for bots"""
    return f"bot_{uuid.uuid4().hex}"


def generate_collection_name(bot_name: str, user_id: int) -> str:
    """Generate a unique ChromaDB collection name"""
    # Remove special characters and spaces, convert to lowercase
    clean_name = "".join(c.lower() for c in bot_name if c.isalnum() or c == "_")
    return f"bot_{user_id}_{clean_name}_{uuid.uuid4().hex[:8]}"
