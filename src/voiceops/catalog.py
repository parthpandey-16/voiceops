"""Product catalog and pincode lookups (data/products.csv, data/pincodes.csv)."""
from __future__ import annotations

import re

from .kb import load_pincodes, load_products, render_product
from .textnorm import find_skus, normalize_numbers

_STOP = {"the", "and", "with", "for", "of", "pack", "ml", "kg", "cm", "inch", "litre"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 and t not in _STOP}


def find_products(text: str, limit: int = 4) -> list[dict]:
    """Products named in a message, by SKU or by (fuzzy) product title."""
    by_sku = {p["sku"]: p for p in load_products()}
    hits = [by_sku[s] for s in find_skus(text) if s in by_sku]
    if hits:
        return hits[:limit]
    words = _tokens(normalize_numbers(text))
    scored = []
    for product in load_products():
        base, _, colour = product["title"].partition(",")
        need = _tokens(base)
        if not need:
            continue
        matched = len(need & words)
        if matched / len(need) >= 0.75:
            colour = colour.strip().lower()
            bonus = 0.5 if colour and colour in text.lower() else 0
            scored.append((matched + bonus, product))
    if not scored:
        return []
    best = max(s for s, _ in scored)
    # The most specific title wins; a colourless mention ("Sootra Canvas Tote
    # Bag") keeps every colour variant, since they share the same policy facts.
    return [p for s, p in sorted(scored, key=lambda x: -x[0]) if s >= best - 0.25][:limit]


def get_product(query: str) -> dict:
    matches = find_products(query)
    if not matches:
        return {"found": False, "message": "No catalog product matches that name or SKU."}
    return {"found": True, "products": [
        {**p, "evidence_id": f"products.csv:{p['sku']}", "summary": render_product(p)}
        for p in matches]}


def check_pincode(pincode: str) -> dict:
    pin = re.sub(r"\D", "", normalize_numbers(str(pincode)))
    row = load_pincodes().get(pin)
    if not row:
        return {"pincode": pin, "listed": False,
                "message": "Pincode not in the serviceable pincode table (DEL-5)."}
    return {"pincode": pin, "listed": True, **row}
