from pydantic import BaseModel
from typing import Optional, List

# ========== AUTH ==========

class RequestOtp(BaseModel):
    email: str


class VerifyOtp(BaseModel):
    email: str
    otp: str


class CredLogin(BaseModel):
    user_id: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    user: dict
    password: Optional[str] = None


# ========== APPOINTMENTS ==========

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


# ========== ADMIN LOGIN ==========

class AdminLogin(BaseModel):
    username: str
    password: str
