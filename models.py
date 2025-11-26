from sqlalchemy import Column, Integer, String, Date, Time
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    password_hash = Column(String)
    role = Column(String, default="user")


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    patient_name = Column(String)
    email = Column(String)
    doctor_name = Column(String)
    appointment_date = Column(Date)
    appointment_time = Column(Time)
    message = Column(String)
    status = Column(String, default="pending")
