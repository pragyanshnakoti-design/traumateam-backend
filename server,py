# server.py - FINAL CLEAN & FULLY WORKING BACKEND WITH RESEND

import os
import resend
import random
from datetime import datetime, timedelta

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr

from sqlalchemy import Column, Integer, String, DateTime, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from passlib.hash import bcrypt
from dotenv import load_dotenv
import jwt

# --------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES (supports quotes)
# --------------------------------------------------------
load_dotenv()

def get_env(key, default=None):
    """Load .env values safely and strip quotes."""
    val = os.getenv(key, default)
    if val is None:
        return default
    return val.strip().replace('"', '').replace("'", "")

RESEND_API_KEY = get_env("RESEND_API_KEY")
RESEND_FROM = get_env("RESEND_FROM", "Noreplytrauma@resend.dev")
ADMIN_USER = get_env("ADMIN_USER", "admin")
ADMIN_PASS = get_env("ADMIN_PASS", "admin123")
JWT_SECRET = get_env("JWT_SECRET", "cyberpunk2077")

if not RESEND_API_KEY:
    raise RuntimeError("Missing RESEND_API_KEY in .env")

resend.api_key = RESEND_API_KEY

# --------------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------------
DATABASE_URL = "sqlite:///./trauma_local.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------------
# DATABASE MODELS
# --------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True)
    username = Column(String)
    password = Column(String)

class OTP(Base):
    __tablename__ = "otp"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String)
    otp = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    doctor = Column(String)
    appointment_date = Column(String)
    appointment_time = Column(String)
    message = Column(Text)
    status = Column(String, default="PENDING")

Base.metadata.create_all(bind=engine)

# --------------------------------------------------------
# EMAIL SENDER (RESEND)
# --------------------------------------------------------
def send_email(to, subject, html):
    resend.Emails.send({
        "from": RESEND_FROM,
        "to": [to],
        "subject": subject,
        "html": html
    })

# --------------------------------------------------------
# AUTH HELPERS
# --------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_token(data):
    data["exp"] = datetime.utcnow() + timedelta(hours=8)
    return jwt.encode(data, JWT_SECRET, algorithm="HS256")

def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except:
        return None

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")

    user = db.query(User).filter(User.id == payload["user_id"]).first()
    if not user:
        raise HTTPException(401, "User not found")

    return user

# --------------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------------
class OTPRequest(BaseModel):
    email: EmailStr

class Register(BaseModel):
    email: EmailStr
    otp: str
    username: str
    password: str

class Login(BaseModel):
    email: EmailStr
    password: str

class AppointmentCreate(BaseModel):
    doctor: str
    appointment_date: str
    appointment_time: str
    message: str | None = None

class AdminStatus(BaseModel):
    status: str

# --------------------------------------------------------
# FASTAPI APP
# --------------------------------------------------------
app = FastAPI(title="Trauma Resend Backend (Final Version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "OK", "message": "Trauma Backend Running with Resend Successfully"}

# --------------------------------------------------------
# OTP ENDPOINTS
# --------------------------------------------------------
@app.post("/otp/request")
def request_otp(payload: OTPRequest, db: Session = Depends(get_db)):
    code = str(random.randint(1000, 9999))

    db.query(OTP).filter(OTP.email == payload.email).delete()
    db.add(OTP(email=payload.email, otp=code))
    db.commit()

    send_email(payload.email, "Your OTP Code", f"<h1>Your OTP is {code}</h1>")

    return {"message": "OTP sent"}

@app.post("/register")
def register(payload: Register, db: Session = Depends(get_db)):
    record = db.query(OTP).filter(OTP.email == payload.email, OTP.otp == payload.otp).first()

    if not record:
        raise HTTPException(400, "Invalid OTP")

    if datetime.utcnow() - record.created_at > timedelta(minutes=5):
        db.delete(record)
        db.commit()
        raise HTTPException(400, "OTP expired")

    hashed = bcrypt.hash(payload.password)
    user = User(email=payload.email, username=payload.username, password=hashed)

    db.add(user)
    db.delete(record)
    db.commit()

    token = create_token({"user_id": user.id})
    send_email(payload.email, "Welcome to TraumaTeam", "<p>Registration successful!</p>")

    return {"message": "Registered", "token": token, "user_id": user.id}

# --------------------------------------------------------
# LOGIN
# --------------------------------------------------------
@app.post("/login")
def login(payload: Login, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()

    if not user or not bcrypt.verify(payload.password, user.password):
        raise HTTPException(401, "Invalid login credentials")

    token = create_token({"user_id": user.id})
    return {"token": token, "user_id": user.id, "username": user.username}

# --------------------------------------------------------
# USER APPOINTMENTS
# --------------------------------------------------------
@app.post("/appointments")
def create_appointment(
    payload: AppointmentCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    appt = Appointment(
        user_id=user.id,
        doctor=payload.doctor,
        appointment_date=payload.appointment_date,
        appointment_time=payload.appointment_time,
        message=payload.message
    )

    db.add(appt)
    db.commit()

    return {"message": "Appointment created", "id": appt.id}

@app.get("/appointments")
def list_appointments(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Appointment).filter(Appointment.user_id == user.id).all()

# --------------------------------------------------------
# ADMIN PANEL
# --------------------------------------------------------
@app.post("/admin/login")
def admin_login(payload: Login):
    if payload.email != ADMIN_USER or payload.password != ADMIN_PASS:
        raise HTTPException(401, "Invalid admin login")

    token = create_token({"is_admin": True})
    return {"token": token}

@app.get("/admin/appointments")
def admin_all(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    data = decode_token(token)
    if not data or not data.get("is_admin"):
        raise HTTPException(403, "Admins only")

    return db.query(Appointment).all()

@app.patch("/admin/appointments/{id}")
def admin_update(
    id: int,
    payload: AdminStatus,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    data = decode_token(token)
    if not data or not data.get("is_admin"):
        raise HTTPException(403, "Only admin allowed")

    appt = db.query(Appointment).filter(Appointment.id == id).first()
    if not appt:
        raise HTTPException(404, "Appointment not found")

    appt.status = payload.status.upper()
    db.commit()

    return {"message": "Updated", "status": appt.status}
