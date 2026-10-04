"""
Authentication and authorization module.
JWT-based auth with role-based access control.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from passlib.context import CryptContext
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.config import settings
from app.db import User, UserRole, get_db

# Use sha256_crypt instead of bcrypt to avoid bcrypt backend issues
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.algorithm)
    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.algorithm])
        return payload
    except JWTError:
        return None


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_roles(*allowed_roles: UserRole):
    """Dependency factory for role-based access control."""
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {current_user.role} not authorized for this action"
            )
        return current_user
    return role_checker


def require_state_scope(current_user: User = Depends(get_current_active_user)) -> User:
    """Ensure state officers only see their state's data."""
    if current_user.role == UserRole.state_officer and not current_user.state:
        raise HTTPException(status_code=400, detail="State officer must have state assigned")
    return current_user


def require_district_scope(current_user: User = Depends(get_current_active_user)) -> User:
    """Ensure district officers only see their district's data."""
    if current_user.role == UserRole.district_officer:
        if not current_user.state or not current_user.district:
            raise HTTPException(status_code=400, detail="District officer must have state and district assigned")
    return current_user


def require_bank_scope(current_user: User = Depends(get_current_active_user)) -> User:
    """Ensure bank nodal officers only see their bank's data."""
    if current_user.role == UserRole.bank_nodal and not current_user.bank:
        raise HTTPException(status_code=400, detail="Bank nodal officer must have bank assigned")
    return current_user


# Role dependency shortcuts
require_admin = require_roles(UserRole.i4c_admin)
require_state_or_admin = require_roles(UserRole.i4c_admin, UserRole.state_officer)
require_district_or_above = require_roles(
    UserRole.i4c_admin, UserRole.state_officer, UserRole.district_officer
)
require_bank_or_admin = require_roles(UserRole.i4c_admin, UserRole.bank_nodal)
require_any_role = require_roles(
    UserRole.i4c_admin, UserRole.state_officer, 
    UserRole.district_officer, UserRole.bank_nodal
)


def filter_by_user_scope(query, model, current_user: User):
    """Filter a query based on user's role and scope."""
    if current_user.role == UserRole.i4c_admin:
        return query  # Admin sees everything
    
    elif current_user.role == UserRole.state_officer:
        # State officer sees only their state
        if hasattr(model, 'state'):
            return query.filter(model.state == current_user.state)
        elif hasattr(model, 'victim_state'):
            return query.filter(model.victim_state == current_user.state)
        elif hasattr(model, 'city'):
            # Filter by cities in their state
            from app.config import CITY_TO_STATE
            allowed_cities = [c for c, s in CITY_TO_STATE.items() if s == current_user.state]
            return query.filter(model.city.in_(allowed_cities))
    
    elif current_user.role == UserRole.district_officer:
        # District officer sees only their district (approximate via city)
        if hasattr(model, 'district'):
            return query.filter(model.district == current_user.district)
        elif hasattr(model, 'victim_city'):
            # For complaints, district ~= victim_city in our synthetic data
            return query.filter(model.victim_city == current_user.district)
        elif hasattr(model, 'city'):
            # For now, district ~= city in our synthetic data
            return query.filter(model.city == current_user.district)
    
    elif current_user.role == UserRole.bank_nodal:
        # Bank nodal sees only their bank's complaints
        if hasattr(model, 'victim_bank'):
            return query.filter(model.victim_bank == current_user.bank)
        elif hasattr(model, 'bank'):
            return query.filter(model.bank == current_user.bank)
    
    return query