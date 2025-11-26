import json
import os
import threading
from typing import List, Dict, Any, Optional

USERS_FILE = "users.json"
APPOINTMENTS_FILE = "appointments.json"

DATA_LOCK = threading.Lock()


def _ensure_file(path: str, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f)


def load_json(path: str, default):
    _ensure_file(path, default)
    with DATA_LOCK:
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default


def save_json(path: str, data):
    with DATA_LOCK:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_users() -> List[Dict[str, Any]]:
    return load_json(USERS_FILE, [])


def save_users(users: List[Dict[str, Any]]):
    save_json(USERS_FILE, users)


def load_appointments() -> List[Dict[str, Any]]:
    return load_json(APPOINTMENTS_FILE, [])


def save_appointments(apps: List[Dict[str, Any]]):
    save_json(APPOINTMENTS_FILE, apps)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    users = load_users()
    for u in users:
        if u.get("email") == email:
            return u
    return None


def get_user_by_user_id(user_id: str) -> Optional[Dict[str, Any]]:
    users = load_users()
    for u in users:
        if u.get("user_id") == user_id:
            return u
    return None


def save_or_update_user(user: Dict[str, Any]) -> Dict[str, Any]:
    users = load_users()
    for i, u in enumerate(users):
        if u.get("user_id") == user.get("user_id"):
            users[i] = user
            save_users(users)
            return user
    users.append(user)
    save_users(users)
    return user


def next_appointment_id() -> int:
    apps = load_appointments()
    if not apps:
        return 1
    return max(a.get("id", 0) for a in apps) + 1
