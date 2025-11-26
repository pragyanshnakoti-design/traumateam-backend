from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

class Token(BaseModel):
    access_token: str
    token_type: str

class AppointmentCreate(BaseModel):
    name: str
    email: EmailStr
    phone: str
    date: str
    time: str
    description: str

class AppointmentOut(BaseModel):
    id: int
    name: str
    email: str
    phone: str
    date: str
    time: str
    description: str
    created_at: datetime

    class Config:
        from_attributes = True
