from __future__ import annotations

import json
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

_initialized = False
_db = None


def get_db():
    global _initialized, _db
    if _initialized:
        return _db

    cred_path = os.environ.get("FORGEX_FIREBASE_CREDENTIALS", "")
    if not cred_path:
        cred_path = str(Path(__file__).resolve().parent.parent.parent.parent / "firebase-credentials.json")

    if not os.path.exists(cred_path):
        raise RuntimeError(
            f"Firebase credentials not found at {cred_path}. "
            "Download your service account JSON from Firebase Console → Project Settings → Service Accounts "
            "and save it as 'firebase-credentials.json' in the project root, "
            "or set FORGEX_FIREBASE_CREDENTIALS env var."
        )

    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    _db = firestore.client()
    _initialized = True
    return _db
