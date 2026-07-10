from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from typing import Literal

from forgex.auth.firebase_config import get_db

Role = Literal["admin", "landlord", "tenant"]
USERS_COL = "users"
PROFILES_COL = "tenant_profiles"

tokens: dict[str, dict] = {}


def _hash_password(password: str) -> str:
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored: str) -> bool:
    salt_hex, key_hex = stored.split(":")
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return key.hex() == key_hex


def create_token(handle: str) -> str:
    token = secrets.token_hex(32)
    tokens[token] = {"handle": handle, "created_at": time.time()}
    return token


def validate_token(token: str) -> dict | None:
    entry = tokens.get(token)
    if entry is None:
        return None
    if time.time() - entry["created_at"] > 2592000:
        tokens.pop(token, None)
        return None
    return entry


def revoke_token(token: str):
    tokens.pop(token, None)


def _user_doc(handle: str) -> dict | None:
    doc = get_db().collection(USERS_COL).document(handle).get()
    return doc.to_dict() if doc.exists else None


def is_handle_taken(handle: str) -> bool:
    return _user_doc(handle) is not None


_HANDLE_RE = re.compile(r"^[a-z0-9_]+$")
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def register_user(handle: str, email: str, password: str, role: Role) -> dict | str:
    handle = handle.lower()
    if is_handle_taken(handle):
        return "handle already exists"
    if len(handle) < 3:
        return "handle must be at least 3 characters"
    if not _HANDLE_RE.match(handle):
        return "handle can only contain lowercase letters, numbers, and underscores"
    if not _EMAIL_RE.match(email):
        return "invalid email format"
    if len(password) < 6:
        return "password must be at least 6 characters"

    db = get_db()
    data = {
        "handle": handle,
        "email": email,
        "password_hash": _hash_password(password),
        "role": role,
        "created_at": time.time(),
    }
    db.collection(USERS_COL).document(handle).set(data)

    if role == "tenant":
        db.collection(PROFILES_COL).document(handle).set({
            "handle": handle,
            "full_name": handle,
            "email": email,
            "phone": "",
            "landlord_handle": "",
            "created_at": time.time(),
        })

    return {"handle": handle, "email": email, "role": role}


def authenticate_user(handle: str, password: str) -> dict | str:
    handle = handle.lower()
    user = _user_doc(handle)
    if user is None:
        return "invalid handle or password"
    if not _verify_password(password, user["password_hash"]):
        return "invalid handle or password"
    token = create_token(handle)
    return {"token": token, "handle": handle, "role": user["role"], "email": user["email"]}


def get_user(handle: str) -> dict | None:
    user = _user_doc(handle)
    if user is None:
        return None
    return {"handle": user["handle"], "email": user["email"], "role": user["role"], "created_at": user["created_at"]}


def list_all_users() -> list[dict]:
    docs = get_db().collection(USERS_COL).stream()
    return [
        {"handle": d.id, "email": d.get("email"), "role": d.get("role"), "created_at": d.get("created_at")}
        for d in docs
    ]


def create_tenant_profile(landlord_handle: str, tenant_handle: str, full_name: str, email: str, phone: str = "") -> dict | str:
    tenant_handle = tenant_handle.lower()
    if not _HANDLE_RE.match(tenant_handle):
        return "handle can only contain lowercase letters, numbers, and underscores"
    if is_handle_taken(tenant_handle):
        return "handle already exists"
    if not _EMAIL_RE.match(email):
        return "invalid email format"

    db = get_db()
    batch = db.batch()

    user_ref = db.collection(USERS_COL).document(tenant_handle)
    batch.set(user_ref, {
        "handle": tenant_handle,
        "email": email,
        "password_hash": _hash_password(secrets.token_hex(8)),
        "role": "tenant",
        "created_at": time.time(),
        "landlord_handle": landlord_handle,
    })

    profile_ref = db.collection(PROFILES_COL).document(tenant_handle)
    batch.set(profile_ref, {
        "handle": tenant_handle,
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "landlord_handle": landlord_handle,
        "created_at": time.time(),
    })

    batch.commit()
    return {
        "handle": tenant_handle,
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "landlord_handle": landlord_handle,
    }


def get_landlord_tenants(landlord_handle: str) -> list[dict]:
    docs = get_db().collection(PROFILES_COL).where("landlord_handle", "==", landlord_handle).stream()
    return [{"handle": d.id, **d.to_dict()} for d in docs]


def get_tenant_profile(tenant_handle: str) -> dict | None:
    doc = get_db().collection(PROFILES_COL).document(tenant_handle).get()
    return doc.to_dict() if doc.exists else None
