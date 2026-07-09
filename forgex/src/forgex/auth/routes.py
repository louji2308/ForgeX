from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Header

from forgex.auth.core import (
    authenticate_user,
    create_tenant_profile,
    get_landlord_tenants,
    get_tenant_profile,
    get_user,
    list_all_users,
    register_user,
    revoke_token,
    validate_token,
)
from forgex.auth.schemas import (
    AuthResponse,
    CreateTenantRequest,
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    TenantProfileResponse,
    UserResponse,
)

router = APIRouter(tags=["auth"])


def require_auth(authorization: str = Header(...)) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="invalid authorization header")
    token = authorization[7:]
    entry = validate_token(token)
    if entry is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    user = get_user(entry["handle"])
    if user is None:
        raise HTTPException(status_code=401, detail="user not found")
    return user


def require_role(role: str):
    def dependency(user: dict = Depends(require_auth)):
        if user["role"] != role:
            raise HTTPException(status_code=403, detail=f"requires {role} role")
        return user
    return dependency


@router.post("/auth/register", response_model=RegisterResponse)
def register(req: RegisterRequest):
    result = register_user(req.handle, req.email, req.password, req.role)
    if isinstance(result, str):
        raise HTTPException(status_code=409, detail=result)
    return result


@router.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest):
    result = authenticate_user(req.handle, req.password)
    if isinstance(result, str):
        raise HTTPException(status_code=401, detail=result)
    return result


@router.get("/auth/me", response_model=UserResponse)
def me(user: dict = Depends(require_auth)):
    return user


@router.post("/auth/logout")
def logout(authorization: str = Header(...)):
    if authorization.startswith("Bearer "):
        revoke_token(authorization[7:])
    return {"status": "ok"}


@router.post("/landlords/tenants", response_model=TenantProfileResponse)
def create_tenant(req: CreateTenantRequest, user: dict = Depends(require_role("landlord"))):
    result = create_tenant_profile(
        landlord_handle=user["handle"],
        tenant_handle=req.tenant_handle,
        full_name=req.full_name,
        email=req.email,
        phone=req.phone,
    )
    if isinstance(result, str):
        raise HTTPException(status_code=409, detail=result)
    return result


@router.get("/landlords/tenants", response_model=list[TenantProfileResponse])
def list_tenants(user: dict = Depends(require_role("landlord"))):
    return get_landlord_tenants(user["handle"])


@router.get("/tenants/{tenant_handle}", response_model=TenantProfileResponse)
def get_tenant(tenant_handle: str, user: dict = Depends(require_auth)):
    profile = get_tenant_profile(tenant_handle)
    if profile is None:
        raise HTTPException(status_code=404, detail="tenant not found")
    if user["role"] == "tenant" and user["handle"] != tenant_handle:
        raise HTTPException(status_code=403, detail="cannot view other tenants")
    if user["role"] == "landlord" and profile.get("landlord_handle") != user["handle"]:
        raise HTTPException(status_code=403, detail="not your tenant")
    return profile


@router.get("/admin/users", response_model=list[UserResponse])
def admin_list_users(user: dict = Depends(require_role("admin"))):
    return list_all_users()
