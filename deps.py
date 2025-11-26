from typing import Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

from config import SECRET_KEY, ALGORITHM
from storage import get_user_by_user_id

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub: str = payload.get("sub")
        if sub is None:
            raise credentials_exception
        role: str = payload.get("role", "user")
    except JWTError:
        raise credentials_exception

    if sub == "ADMIN":
        return {"user_id": "ADMIN", "email": "admin@traumateam.local", "role": "admin"}

    user = get_user_by_user_id(sub)
    if not user:
        raise credentials_exception

    user["role"] = role or user.get("role", "user")
    return user


async def get_current_admin(current_user: Dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return current_user
