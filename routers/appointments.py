from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Appointment
from schemas import AppointmentCreate, AppointmentOut
from auth import get_current_user

router = APIRouter(prefix="/api/appointments", tags=["Appointments"])

@router.post("")
def create_appointment(payload: AppointmentCreate, db: Session = Depends(get_db),
                       current_user=Depends(get_current_user)):
    new_app = Appointment(
        patient_name=payload.patient_name,
        email=payload.email,
        doctor_name=payload.doctor_name,
        appointment_date=payload.appointment_date,
        appointment_time=payload.appointment_time,
        message=payload.message,
        status="pending",
        user_id=current_user.id,
    )
    db.add(new_app)
    db.commit()
    db.refresh(new_app)
    return {"message": "Appointment created", "id": new_app.id}


@router.get("/me", response_model=list[AppointmentOut])
def get_my_appointments(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    apps = db.query(Appointment).filter(Appointment.user_id == current_user.id).all()
    return apps
