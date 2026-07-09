from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forgex.auth.core import register_user

try:
    user = register_user("admin", "admin@forgex.ai", "admin123", "admin")
    print(f"Created admin user: @{user['handle']} (role={user['role']})")
except ValueError as e:
    print(f"Admin user already exists: {e}")
except RuntimeError as e:
    print(f"Firebase not configured: {e}")
    print("Tip: Download your service account JSON from Firebase Console and save as firebase-credentials.json")
