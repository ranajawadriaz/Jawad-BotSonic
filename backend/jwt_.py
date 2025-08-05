from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from config import settings
from models import User
from schemas import TokenData


# Password hashing
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception as e:
    # Fallback for bcrypt version issues
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

# JWT token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT refresh token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str, token_type: str = "access") -> Optional[TokenData]:
    """Verify a JWT token and return token data"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        
        # Check token type
        if payload.get("type") != token_type:
            return None
        
        user_id: int = payload.get("sub")
        email: str = payload.get("email")
        
        if user_id is None:
            return None
        
        return TokenData(user_id=user_id, email=email)
    
    except JWTError:
        return None


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user with email and password"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def get_current_user_from_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from Authorization header token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token_data = verify_token(credentials.credentials, "access")
        if token_data is None:
            raise credentials_exception
        return token_data
    except Exception:
        raise credentials_exception


def get_current_user_from_cookie(access_token: Optional[str] = Cookie(None)):
    """Get current user from cookie token"""
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    token_data = verify_token(access_token, "access")
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return token_data


def get_current_user(db: Session, token_data: TokenData) -> User:
    """Get current user from database using token data"""
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return user


def require_authentication(
    db: Session,
    token_data: TokenData = Depends(get_current_user_from_token)
) -> User:
    """Dependency to require authentication via Authorization header"""
    return get_current_user(db, token_data)


def get_current_user_flexible(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    access_token_cookie: Optional[str] = Cookie(None, alias="access_token")
) -> TokenData:
    """Get token data from either Authorization header or cookie"""
    token_data = None
    
    # Try Authorization header first
    if credentials:
        try:
            token_data = verify_token(credentials.credentials, "access")
        except Exception:
            pass
    
    # If no header auth, try cookie
    if not token_data and access_token_cookie:
        try:
            token_data = verify_token(access_token_cookie, "access")
        except Exception:
            pass
    
    # If no valid token found
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return token_data


def require_authentication_flexible(
    token_data: TokenData = Depends(get_current_user_flexible)
) -> TokenData:
    """Dependency to require authentication via either header or cookie"""
    return token_data


def require_authentication_cookie(
    db: Session,
    token_data: TokenData = Depends(get_current_user_from_cookie)
) -> User:
    """Dependency to require authentication via cookie"""
    return get_current_user(db, token_data)


def create_token_pair(user: User) -> dict:
    """Create both access and refresh tokens for a user"""
    access_token_expires = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    access_token = create_access_token(
        data={"sub": str(user.id), "email": user.email},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": str(user.id), "email": user.email},
        expires_delta=refresh_token_expires
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    }


def refresh_access_token(refresh_token: str) -> Optional[str]:
    """Create a new access token using a refresh token"""
    token_data = verify_token(refresh_token, "refresh")
    if token_data is None:
        return None
    
    new_access_token = create_access_token(
        data={"sub": str(token_data.user_id), "email": token_data.email}
    )
    
    return new_access_token


def verify_bot_api_token(api_token: str, db: Session):
    """Verify bot API token for REST API access"""
    from .models import Bot
    
    bot = db.query(Bot).filter(Bot.api_token == api_token, Bot.is_active == True).first()
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token"
        )
    return bot


class BotAPITokenAuth:
    """Authentication dependency for bot API endpoints"""
    
    def __init__(self, db: Session, api_token: str = Depends(lambda: None)):
        self.bot = verify_bot_api_token(api_token, db)
    
    @property
    def bot_id(self) -> int:
        return self.bot.id
    
    @property
    def collection_name(self) -> str:
        return self.bot.collection_name
