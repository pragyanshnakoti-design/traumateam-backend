from pydantic import BaseModel
from typing import Optional, List

# ---------- AUTH ----------
class RequestOtp(BaseModel):
    email: str

class VerifyOtp(BaseModel):
    email: str
    otp: str

# For auth.py compatibility
class OTPVerify(BaseModel):
    email: str
    otp: str

class UserCreate(BaseModel):
    email: str
    full_name: Optional[str] = None

class CredLogin(BaseModel):
    user_id: str
    password: str

class TokenUser(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    user: TokenUser


# ---------- APPOINTMENTS ----------
class AppointmentCreate(BaseModel):
    patient_name: str
    email: Optional[str] = None
    doctor_name: str
    appointment_date: str
    appointment_time: str
    message: Optional[str] = None

class AppointmentOut(BaseModel):
    id: int
    user_id: str
    patient_name: str
    email: str
    doctor_name: str
    appointment_date: str
    appointment_time: str
    message: str
    status: str
    created_at: str


# ---------- ADMIN ----------
class AdminLogin(BaseModel):
    username: str
    password: str
