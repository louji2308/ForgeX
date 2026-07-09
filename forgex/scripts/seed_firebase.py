from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forgex.auth.firebase_config import get_db

db = get_db()

# Create users collection with an admin seed
from forgex.auth.core import register_user, list_all_users

try:
    result = register_user("admin", "admin@forgex.ai", "admin123", "admin")
    print(f"Seeded admin: @{result['handle']}")
except ValueError:
    print("Admin already exists")

users = list_all_users()
print(f"\nTotal users in Firestore: {len(users)}")
for u in users:
    print(f"  @{u['handle']} — {u['role']} ({u['email']})")
