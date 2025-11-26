from datetime import datetime, timezone
from typing import Optional, Dict, Any

from jose import jwt
from passlib.context import CryptContext

from config import SECRET_KEY, ALGORITHM, access_token_expires_delta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta=None) -> str:
    to_encode = data.copy()
    expires_delta = expires_delta or access_token_expires_delta()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
