from __future__ import annotations

import logging
import re
from typing import Any

from forgex.logging_setup import get_logger

logger = get_logger(__name__)

NARRATIVE_TEMPLATE_FALLBACK = (
    "{tenant_name}'s churn risk is {risk_pct:.0f}%. The biggest factor is "
    "{top_driver_label} ({top_driver_direction}). Consider addressing this first."
)


def _build_narrative_prompt(
    tenant_name: str, risk_pct: float, drivers: list[dict]
) -> str:
    drivers_text = "\n".join(
        f"- {d['label']}: {'increases' if d['direction'] == 'increases_risk' else 'decreases'} risk "
        f"(SHAP value: {d['shap_value']:.3f})"
        for d in drivers
    )
    return (
        f"Tenant {tenant_name} has a {risk_pct:.0f}% probability of not renewing their lease. "
        f"The top risk factors are:\n{drivers_text}\n\n"
        f"Write a brief, empathetic narrative explaining this tenant's situation "
        f"to a property manager. Include the specific risk percentage ({risk_pct:.0f}%) "
        f"in your response. Suggest one actionable step. Keep it under 3 sentences."
    )


def _narrative_is_grounded(text: str, risk_pct: float) -> bool:
    """Cheap grounding check: the risk number the model states out loud
    must be within 1 point of the real number. Doesn't catch every
    hallucination, but catches the most embarrassing one."""
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return any(abs(float(n) - risk_pct) <= 1.0 for n in numbers)


def generate_narrative(
    tenant_name: str,
    risk_pct: float,
    drivers: list[dict],
    llm_client: Any | None = None,
    llm_fn: Any | None = None,
    max_retries: int = 1,
    timeout_s: float = 30.0,
) -> dict:
    """Never let a flaky LLM call take down the risk explanation — the
    numeric SHAP output must always render even if the sentence doesn't."""
    if not drivers:
        raise ValueError("generate_narrative called with an empty drivers list")

    if llm_client is not None or llm_fn is not None:
        prompt = _build_narrative_prompt(tenant_name, risk_pct, drivers)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                if llm_fn is not None:
                    text = llm_fn(prompt, timeout_s=timeout_s)
                else:
                    response = llm_client.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=150,
                        messages=[{"role": "user", "content": prompt}],
                        timeout=timeout_s,
                    )
                    text = response.content[0].text.strip()
                if not _narrative_is_grounded(text, risk_pct):
                    raise ValueError(
                        "Narrative failed grounding check — model may have invented a number"
                    )
                return {"narrative": text, "source": "llm", "attempt": attempt}
            except Exception as e:
                last_error = e
                logger.warning(f"Narrative attempt {attempt} failed: {e}")

        logger.error(
            f"All narrative attempts failed, using template fallback: {last_error}"
        )

    top = drivers[0]
    return {
        "narrative": NARRATIVE_TEMPLATE_FALLBACK.format(
            tenant_name=tenant_name,
            risk_pct=risk_pct,
            top_driver_label=top["label"],
            top_driver_direction=top["direction"].replace("_", " "),
        ),
        "source": "template_fallback",
        "attempt": None,
    }
