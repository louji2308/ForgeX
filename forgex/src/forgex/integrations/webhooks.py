from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any, Callable

from forgex.errors import WebhookAuthError
from forgex.logging_setup import get_logger

logger = get_logger(__name__)

_processed_keys: set[str] = set()


def verify_webhook_signature(
    payload_bytes: bytes,
    signature_header: str,
    secret: str,
) -> None:
    if not signature_header:
        raise WebhookAuthError("Missing X-Signature header")
    expected = hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise WebhookAuthError(
            "Signature mismatch — payload tampered with, or secret is wrong"
        )


async def already_processed(idempotency_key: str) -> bool:
    return idempotency_key in _processed_keys


async def mark_processed(idempotency_key: str) -> None:
    _processed_keys.add(idempotency_key)


async def enqueue_for_processing(
    provider: str,
    payload: dict[str, Any],
    idempotency_key: str | None,
) -> None:
    logger.info(
        f"Webhook from {provider} accepted"
        f"{' (idempotency: ' + idempotency_key + ')' if idempotency_key else ''}"
        f": {json.dumps(payload)[:200]}..."
    )
    if idempotency_key:
        await mark_processed(idempotency_key)


class WebhookRouter:
    """Routes incoming PM-platform webhooks to the correct handler
    with signature verification and idempotency."""

    def __init__(self, secrets: dict[str, str] | None = None):
        self.secrets = secrets or {}
        self._handlers: dict[str, Callable] = {}

    def register(self, provider: str, handler: Callable) -> None:
        self._handlers[provider] = handler

    async def receive(
        self,
        provider: str,
        payload_bytes: bytes,
        signature_header: str,
        idempotency_key: str | None = None,
    ) -> dict[str, str]:
        if provider not in self.secrets:
            raise WebhookAuthError(f"No configured secret for provider '{provider}'")

        verify_webhook_signature(
            payload_bytes, signature_header, self.secrets[provider]
        )

        if idempotency_key and await already_processed(idempotency_key):
            return {"status": "duplicate_ignored"}

        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError:
            raise ValueError("Malformed JSON payload")

        if provider in self._handlers:
            await self._handlers[provider](payload)

        await enqueue_for_processing(provider, payload, idempotency_key)
        return {"status": "accepted"}


webhook_router = WebhookRouter()
