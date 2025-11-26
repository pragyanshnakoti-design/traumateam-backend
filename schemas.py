from typing import Optional, List, Dict, Any

from pydantic import BaseModel, EmailStr


class RequestOtp(BaseModel):
    email: EmailStr


class VerifyOtp(BaseModel):
    email: EmailStr
    otp: str


class CredLogin(BaseModel):
    user_id: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]
    password: Optional[str] = None  # temp password (for OTP flow)


class AppointmentCreate(BaseModel):
    patient_name: str
    email: Optional[EmailStr] = None
    doctor_name: str
    appointment_date: str  # "YYYY-MM-DD"
    appointment_time: str  # "HH:MM"
    message: Optional[str] = None


class AppointmentOut(BaseModel):
    id: int
    patient_name: str
    email: Optional[EmailStr]
    doctor_name: str
    appointment_date: str
    appointment_time: str
    message: Optional[str]
    status: str
    created_at: str


class AdminLogin(BaseModel):
    username: str
    password: str
