from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


# User Schemas
class UserBase(BaseModel):
    full_name: str
    email: EmailStr
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    country: Optional[str] = None
    monthly_queries: Optional[str] = None
    budget: Optional[str] = None


class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    country: Optional[str] = None
    subscription_plan: Optional[str] = None


class UserResponse(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    subscription_plan: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False


# Demo Request Schemas
class DemoRequestCreate(BaseModel):
    first_name: str
    last_name: str
    work_email: EmailStr
    company_name: str
    job_title: str
    country: str
    monthly_queries: str
    budget: str
    additional_notes: Optional[str] = None


class DemoRequestResponse(DemoRequestCreate):
    id: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# Bot Schemas
class BotBase(BaseModel):
    name: str
    description: Optional[str] = None


class BotCreate(BotBase):
    pass


class BotUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    widget_name: Optional[str] = None
    accent_color: Optional[str] = None
    widget_icon: Optional[str] = None
    widget_position: Optional[str] = None
    input_placeholder: Optional[str] = None
    response_length: Optional[str] = None
    show_sources: Optional[bool] = None
    rate_limit_per_hour: Optional[int] = None
    working_hours_enabled: Optional[bool] = None
    working_hours_config: Optional[Dict[str, Any]] = None
    user_form_enabled: Optional[bool] = None
    user_form_fields: Optional[List[Dict[str, Any]]] = None
    email_notifications_enabled: Optional[bool] = None
    support_email: Optional[str] = None
    transcript_email: Optional[str] = None


class BotResponse(BotBase):
    id: int
    collection_name: str
    api_token: str
    widget_name: str
    accent_color: str
    widget_position: str
    input_placeholder: str
    response_length: str
    show_sources: bool
    rate_limit_per_hour: int
    working_hours_enabled: bool
    working_hours_config: Optional[Dict[str, Any]]
    user_form_enabled: bool
    user_form_fields: Optional[List[Dict[str, Any]]]
    email_notifications_enabled: bool
    support_email: Optional[str]
    transcript_email: Optional[str]
    total_conversations: int
    total_messages: int
    is_active: bool
    created_at: datetime
    owner_id: int
    
    class Config:
        from_attributes = True


# Guideline Schemas
class GuidelineCreate(BaseModel):
    title: str
    content: str


class GuidelineUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    is_active: Optional[bool] = None


class GuidelineResponse(GuidelineCreate):
    id: int
    is_active: bool
    created_at: datetime
    bot_id: int
    
    class Config:
        from_attributes = True


# Starter Question Schemas
class StarterQuestionCreate(BaseModel):
    question: str
    order_index: Optional[int] = 0


class StarterQuestionUpdate(BaseModel):
    question: Optional[str] = None
    order_index: Optional[int] = None
    is_active: Optional[bool] = None


class StarterQuestionResponse(StarterQuestionCreate):
    id: int
    is_active: bool
    created_at: datetime
    bot_id: int
    
    class Config:
        from_attributes = True


# Source Schemas
class SourceCreate(BaseModel):
    source_type: str  # file, url
    source_name: str
    source_url: Optional[str] = None


class SourceResponse(SourceCreate):
    id: int
    file_size: Optional[int]
    processing_status: str
    error_message: Optional[str]
    chunk_count: int
    created_at: datetime
    processed_at: Optional[datetime]
    bot_id: int
    
    class Config:
        from_attributes = True


# Chat Schemas
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_identifier: Optional[str] = None
    user_form_data: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    bot_response: str
    session_id: str
    sources_used: Optional[List[Dict[str, Any]]] = None
    response_time_ms: Optional[int] = None


class ConversationResponse(BaseModel):
    id: int
    session_id: str
    user_identifier: Optional[str]
    user_form_data: Optional[Dict[str, Any]]
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    bot_id: int
    
    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: int
    content: str
    message_type: str
    sources_used: Optional[List[Dict[str, Any]]]
    response_time_ms: Optional[int]
    created_at: datetime
    conversation_id: int
    
    class Config:
        from_attributes = True


# URL Processing Schema
class URLInput(BaseModel):
    url: str
    
    @validator('url')
    def validate_url(cls, v):
        if not v.strip():
            raise ValueError('URL cannot be empty')
        return v.strip()


# File Upload Response
class FileUploadResponse(BaseModel):
    message: str
    filename: str
    file_size: int
    source_id: int
    processing_status: str


# Analytics Schemas
class BotAnalytics(BaseModel):
    total_conversations: int
    total_messages: int
    messages_today: int
    messages_this_week: int
    messages_this_month: int
    average_response_time_ms: Optional[float]
    top_sources_used: List[Dict[str, Any]]
    conversation_trends: List[Dict[str, Any]]


class UserAnalytics(BaseModel):
    total_bots: int
    total_conversations: int
    total_messages: int
    subscription_plan: str
    usage_this_month: int
    plan_limits: Dict[str, Any]


# Token Schemas
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None


# API Response Schemas
class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Embedding Configuration Schema
class EmbeddingConfig(BaseModel):
    bot_id: int
    collection_name: str
    iframe_url: str
    javascript_snippet: str
    api_endpoint: str
    api_token: str
