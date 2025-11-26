from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Appointment
from schemas import AdminLogin, TokenResponse
from auth import create_access_token
from config import ADMIN_USERNAME, ADMIN_PASSWORD

router = APIRouter(tags=["Admin"])

@router.post("/api/admin/login", response_model=TokenResponse)
def admin_login(payload: AdminLogin):
    if payload.username != ADMIN_USERNAME or payload.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")

    token = create_access_token({"sub": "ADMIN", "role": "admin"})
    return TokenResponse(
        access_token=token,
        user={"user_id": "ADMIN", "email": "admin@traumateam.com", "full_name": "Admin"},
    )


@router.get("/api/admin/appointments")
def all_appointments(db: Session = Depends(get_db)):
    return db.query(Appointment).all()
