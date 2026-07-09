from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    handle: str = Field(..., min_length=3, max_length=30, pattern=r"^[a-z0-9_]+$")
    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    password: str = Field(..., min_length=6)
    role: Literal["admin", "landlord", "tenant"] = "landlord"


class LoginRequest(BaseModel):
    handle: str
    password: str


class AuthResponse(BaseModel):
    token: str
    handle: str
    role: str
    email: str


class UserResponse(BaseModel):
    handle: str
    email: str
    role: str
    created_at: float


class RegisterResponse(BaseModel):
    handle: str
    email: str
    role: str


class CreateTenantRequest(BaseModel):
    tenant_handle: str = Field(..., min_length=3, max_length=30, pattern=r"^[a-z0-9_]+$")
    full_name: str = Field(..., min_length=1)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    phone: str = ""


class TenantProfileResponse(BaseModel):
    handle: str
    full_name: str
    email: str
    phone: str
    landlord_handle: str
