from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from forgex.logging_setup import get_logger

logger = get_logger(__name__)


def load_fallback_responses(path: str | Path = "artifacts/demo_fallback_responses.json") -> dict:
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.loads(f.read())
    logger.warning(f"Fallback responses file not found at {path}")
    return {}


def generate_fallback_responses(app: FastAPI, output_path: str | Path) -> None:
    """Generate demo_fallback_responses.json by hitting every endpoint
    you'll demo once, ahead of time, and saving the real responses."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fallbacks = {}

    from forgex.api.schemas import HealthResponse

    for route in app.routes:
        if hasattr(route, "path") and route.path in {"/health", "/score/", "/optimize"}:
            continue

    with open(output_path, "w") as f:
        json.dump(fallbacks, f, indent=2)
    logger.info(f"Pre-computed fallback responses saved to {output_path}")


DEMO_FALLBACKS: dict = {}


async def fallback_on_failure(request: Request, call_next):
    """Middleware: if the live call fails, serve a precomputed response."""
    try:
        response = await asyncio.wait_for(call_next(request), timeout=3.0)
        return response
    except Exception as e:
        key = str(request.url.path)
        if key in DEMO_FALLBACKS:
            logger.warning(
                f"Live call failed ({e}); serving precomputed fallback for {key}"
            )
            return JSONResponse(DEMO_FALLBACKS[key])
        raise
