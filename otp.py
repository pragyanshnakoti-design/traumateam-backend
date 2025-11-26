import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from fastapi import HTTPException

from email_utils import send_otp_email

# In-memory store: { email: { "otp": str, "expires_at": datetime, "temp_password": str } }
OTP_STORE: Dict[str, Dict[str, Any]] = {}


def generate_user_id() -> str:
    year = datetime.now().year
    suffix = "".join(
        secrets.choice(string.ascii_uppercase + string.digits) for _ in range(4)
    )
    return f"TTI-{year}-{suffix}"


def generate_temp_password(length: int = 10) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def create_and_send_otp(email: str) -> None:
    otp = "".join(secrets.choice(string.digits) for _ in range(6))
    temp_password = generate_temp_password()

    OTP_STORE[email] = {
        "otp": otp,
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
        "temp_password": temp_password,
    }

    # Send email
    send_otp_email(email, otp, temp_password)


def consume_otp(email: str, received_otp: str) -> Dict[str, Any]:
    record = OTP_STORE.get(email)
    if not record:
        raise HTTPException(status_code=400, detail="No OTP requested for this email.")

    if datetime.now(timezone.utc) > record["expires_at"]:
        OTP_STORE.pop(email, None)
        raise HTTPException(status_code=400, detail="OTP expired. Request a new one.")

    if record["otp"] != received_otp:
        raise HTTPException(status_code=400, detail="Invalid OTP.")

    # Valid
    OTP_STORE.pop(email, None)
    return record
