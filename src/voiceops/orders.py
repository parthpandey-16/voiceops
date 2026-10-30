"""Bridge to the dataset's mock tools (tools/tools.py) plus neutral derived facts."""
from __future__ import annotations

import importlib.util
import re
import sys
from functools import lru_cache

from . import dates
from .config import settings
from .kb import load_products

# Never put verification secrets or contact details in front of the LLM.
_HIDDEN = {"delivery_pincode", "phone", "email", "phone_last4", "customer_id"}


@lru_cache(maxsize=1)
def _tools():
    path = settings.dataset_dir / "tools" / "tools.py"
    spec = importlib.util.spec_from_file_location("zentromart_tools", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["zentromart_tools"] = module
    spec.loader.exec_module(module)
    return module


def lookup_order(order_id: str, verification_value: str | None = None) -> dict:
    return _tools().lookup_order(order_id, verification_value=verification_value)


def create_ticket(summary: str, order_id: str | None = None, reason: str | None = None,
                  intent: str | None = None) -> dict:
    return _tools().create_ticket(summary=summary, order_id=order_id, reason=reason, intent=intent)


def _derived(order: dict) -> dict:
    """Day counts the agent needs, computed exactly; the policy decision stays with the agent."""
    today = settings.today
    out = {"today": today}
    if order.get("delivered_on"):
        out["calendar_days_since_delivery"] = dates.calendar_days_between(order["delivered_on"], today)
    if order.get("promised_date"):
        out["promised_date_passed"] = order["promised_date"] < today
    for prefix in ("cancellation_refund", "rto_refund", "return_refund"):
        start = order.get(f"{prefix}_initiated_on")
        if start:
            out[f"{prefix}_business_days_since_initiation"] = dates.business_days_between(start, today)
    if order.get("return_pickup_on"):
        out["return_business_days_since_pickup"] = dates.business_days_between(
            order["return_pickup_on"], today)
    return out


def safe_order_view(order: dict) -> dict:
    """The verified order as the LLM sees it: no secrets, nulls dropped, catalog joined."""
    products = {p["sku"]: p for p in load_products()}
    view = {k: v for k, v in order.items()
            if v is not None and k not in _HIDDEN and k not in ("found", "verified", "items")}
    items = []
    for item in order.get("items", []):
        p = products.get(item.get("sku"), {})
        items.append({
            "line_no": item.get("line_no"), "sku": item.get("sku"), "title": item.get("title"),
            "qty": item.get("qty"), "unit_price_inr": item.get("unit_price_inr"),
            "return_category": item.get("return_category"),
            "fulfilled_by": p.get("fulfilled_by"), "assured_badge": p.get("assured"),
            "brand_warranty_months": p.get("warranty_months"),
        })
    view["items"] = items
    view["derived"] = _derived(order)
    return view


def evidence_for_items(order: dict) -> list[str]:
    return [f"products.csv:{i['sku']}" for i in order.get("items", []) if i.get("sku")]


_SECRET_PATTERNS = [r"\+91[\s-]?\d[\d\s-]{8,}", r"[\w.]+@example\.com"]


def scrub(text: str, secrets: list[str]) -> str:
    """Remove verification values, phone numbers and emails from outgoing text."""
    for secret in secrets:
        if secret:
            text = re.sub(rf"(?<!\d){re.escape(secret)}(?!\d)", "[redacted]", text)
    for pattern in _SECRET_PATTERNS:
        text = re.sub(pattern, "[redacted]", text)
    return text
