"""The ZentroMart support agent.

BaselineAgent  - Step 1: an LLM with a system prompt and nothing else.
SupportAgent   - Steps 2-4: hybrid RAG over the KB, catalog rows, identity-checked
                 order lookup, ticket creation and multi-turn memory. The same
                 object answers typed text and (through voice.py) phone audio.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from . import catalog, dates, orders
from .config import settings
from .kb import chunk_by_id, load_products, render_product
from .llm import LLM, parse_json
from .prompts import AGENT_SYSTEM, BASELINE_SYSTEM, QUERY_REWRITE, VOICE_RULES
from .retriever import Retriever
from .textnorm import find_order_id, find_verification_value, has_devanagari

log = logging.getLogger(__name__)

DECISIONS = {"answer", "clarify", "escalate", "refuse"}
# Agent guides and the KB meta rules are instructions the agent must always
# follow, so they sit in the system prompt rather than depending on retrieval.
ALWAYS_ON = ("STY", "IDV", "ESC", "KB")


def _agent_guides() -> str:
    return "\n\n".join(c.render() for sid, c in chunk_by_id().items()
                       if sid.split("-")[0] in ALWAYS_ON and sid != "KB-00.4")


def _valid_sources(raw) -> list[str]:
    known = set(chunk_by_id())
    skus = {p["sku"] for p in load_products()}
    out = []
    for item in raw or []:
        sid = str(item).strip().strip("[]")
        if sid.startswith("products.csv:"):
            if sid.split(":", 1)[1] in skus:
                out.append(sid)
        elif sid in known:
            out.append(sid)
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------
class BaselineAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

    def answer(self, question: str) -> dict:
        raw = self.llm.complete(question, system=BASELINE_SYSTEM)
        data = parse_json(raw)
        reply = data.get("reply") or raw
        decision = str(data.get("decision", "answer")).lower()
        return {"reply": reply, "decision": decision if decision in DECISIONS else "answer",
                "sources": [], "retrieved": [], "tool_calls": []}


# ---------------------------------------------------------------------------
# Steps 2-4
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search ZentroMart's policy documents, help articles and FAQs. Use when the "
                       "KNOWLEDGE given this turn does not cover the question.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "English search query"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_product",
        "description": "Look up a catalog product by SKU (e.g. ELE-0012) or product name: return "
                       "category, price, seller, Assured badge, brand warranty months.",
        "parameters": {"type": "object", "properties": {
            "name_or_sku": {"type": "string"}}, "required": ["name_or_sku"]}}},
    {"type": "function", "function": {
        "name": "check_pincode",
        "description": "Check whether ZentroMart delivers to a pincode and whether COD/Express "
                       "are available there.",
        "parameters": {"type": "object", "properties": {
            "pincode": {"type": "string"}}, "required": ["pincode"]}}},
    {"type": "function", "function": {
        "name": "date_calc",
        "description": "Exact date arithmetic. Business days = Mon-Sat excluding ZentroMart "
                       "holidays. Use for every deadline, refund date or overdue check.",
        "parameters": {"type": "object", "properties": {
            "operation": {"type": "string", "enum": [
                "add_business_days", "add_calendar_days", "business_days_between",
                "calendar_days_between"]},
            "start": {"type": "string", "description": "YYYY-MM-DD"},
            "days": {"type": "integer"},
            "end": {"type": "string", "description": "YYYY-MM-DD (default today)"}},
            "required": ["operation", "start"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Create a human-handover ticket. Required before telling a customer a "
                       "human colleague will contact them.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description":
                        "ESC-3 summary: issue, intent, order ID, what the customer said, what "
                        "you checked, reason for escalation. No verification digits."},
            "reason": {"type": "string", "description": "short code, e.g. high_value_damage, "
                       "human_requested, repeat_contact, refund_overdue, fraud, legal, safety"},
            "intent": {"type": "string"}},
            "required": ["summary", "reason"]}}},
]


@dataclass
class ConversationState:
    order_id: str | None = None
    verified: bool = False
    attempts: int = 0
    locked: bool = False
    order: dict | None = None
    pending_value: str | None = None
    secrets: list[str] = field(default_factory=list)
    products: list[dict] = field(default_factory=list)
    tickets: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    turns: int = 0

    def describe(self) -> str:
        if not self.order_id:
            return "No order ID given yet."
        if self.verified:
            return f"Order {self.order_id}: caller VERIFIED. Order details may be shared."
        if self.locked:
            return (f"Order {self.order_id}: verification FAILED twice. Do NOT share any order "
                    "details. Offer general policy help and suggest contacting support from the "
                    "registered phone number (IDV-3).")
        if self.attempts == 1:
            return (f"Order {self.order_id}: the verification value did not match (1 of 2 attempts "
                    "used). Ask the customer to try again with the last 4 phone digits or the "
                    "delivery pincode. Share no details.")
        return (f"Order {self.order_id}: NOT verified yet. Ask for the last 4 digits of the "
                "registered phone OR the delivery pincode. Share no details.")


class SupportAgent:
    def __init__(self, llm: LLM, retriever: Retriever | None = None, voice: bool = False,
                 rewrite: bool = True, k: int | None = None):
        self.llm = llm
        self.retriever = retriever or Retriever(llm)
        self.voice = voice
        self.rewrite = rewrite
        self.k = k or settings.top_k
        self.system = AGENT_SYSTEM.format(agent_guides=_agent_guides(),
                                          voice_rules=VOICE_RULES if voice else "")
        self.state = ConversationState()

    def reset(self):
        self.state = ConversationState()

    # -- identity-checked order handling (deterministic, never left to the LLM)
    def _handle_order(self, message: str, calls: list[dict]):
        st = self.state
        oid = find_order_id(message)
        if oid and oid != st.order_id:
            st.order_id, st.verified, st.attempts, st.locked, st.order = oid, False, 0, False, None
            result = orders.lookup_order(oid)
            calls.append({"tool": "lookup_order", "args": {"order_id": oid},
                          "result": {k: result.get(k) for k in ("verification_required", "message")}})
        value = find_verification_value(message, st.order_id or oid)
        if value and not st.order_id:
            st.pending_value = value  # remember it until we know the order ID
            return
        if not value and st.pending_value and st.order_id and not st.verified:
            value, st.pending_value = st.pending_value, None
        if value and st.order_id and not st.verified and not st.locked:
            st.secrets.append(value)
            result = orders.lookup_order(st.order_id, verification_value=value)
            ok = bool(result.get("verified"))
            calls.append({"tool": "lookup_order",
                          "args": {"order_id": st.order_id, "verification_value": "<redacted>"},
                          "result": {"verified": ok}})
            if ok:
                st.verified, st.order = True, result
            else:
                st.attempts += 1
                st.locked = st.attempts >= 2

    # -- retrieval ----------------------------------------------------------
    def _queries(self, message: str) -> tuple[list[str], str]:
        if not self.rewrite:
            return [message], ""
        history = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in self.state.history[-6:])
        try:
            data = parse_json(self.llm.complete(
                QUERY_REWRITE.format(history=history or "(none)", message=message), json_mode=True))
        except Exception as exc:  # noqa: BLE001
            log.warning("query rewrite failed: %s", exc)
            data = {}
        query = data.get("query") or ""
        return [q for q in (query, message) if q], data.get("language", "")

    def _context(self, message: str, hits, products) -> str:
        st = self.state
        parts = [f"TURN: {'first message of the conversation' if st.turns == 0 else f'turn {st.turns + 1}'}",
                 f"ORDER STATE: {st.describe()}"]
        if st.verified and st.order:
            view = orders.safe_order_view(st.order)
            parts.append("VERIFIED ORDER (from lookup_order):\n" + json.dumps(view, ensure_ascii=False))
        if products:
            parts.append("CATALOG ROWS:\n" + "\n".join(render_product(p) for p in products))
        if st.tickets:
            parts.append("TICKETS ALREADY CREATED: " + ", ".join(t["ticket_id"] for t in st.tickets))
        parts.append("KNOWLEDGE (retrieved sections):\n" + "\n\n".join(h.chunk.render() for h in hits))
        return "\n\n".join(parts) + f"\n\nCUSTOMER MESSAGE: {message}"

    # -- tool execution -----------------------------------------------------
    def _run_tool(self, name: str, args: dict, calls: list[dict], extra_hits: list) -> dict:
        st = self.state
        if name == "search_knowledge_base":
            hits = self.retriever.search(args.get("query", ""), k=4)
            extra_hits.extend(hits)
            result = {"sections": [h.chunk.render() for h in hits]}
        elif name == "get_product":
            result = catalog.get_product(args.get("name_or_sku", ""))
        elif name == "check_pincode":
            result = catalog.check_pincode(args.get("pincode", ""))
        elif name == "date_calc":
            try:
                result = dates.date_calc(args.get("operation", ""), args.get("start", ""),
                                         args.get("days"), args.get("end"))
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
        elif name == "create_ticket":
            summary = orders.scrub(args.get("summary", ""), st.secrets)
            result = orders.create_ticket(summary, order_id=st.order_id,
                                          reason=args.get("reason"), intent=args.get("intent"))
            if result.get("created"):
                st.tickets.append(result)
            args = {**args, "summary": summary, "order_id": st.order_id}
        else:
            result = {"error": f"unknown tool {name}"}
        calls.append({"tool": name, "args": args, "result": result})
        return result

    def _fallback_ticket(self, message: str, calls: list[dict]) -> dict:
        st = self.state
        said = " | ".join(m["content"][:160] for m in st.history if m["role"] == "user")
        summary = orders.scrub(
            f"Handover requested. Latest customer message: {message[:300]}. Earlier: {said or 'none'}. "
            f"Order: {st.order_id or 'not given'} (verified={st.verified}). "
            "Escalation required under ESC-1; agent could not resolve within policy.", st.secrets)
        result = orders.create_ticket(summary, order_id=st.order_id, reason="escalation")
        calls.append({"tool": "create_ticket", "args": {"summary": summary, "order_id": st.order_id,
                                                         "reason": "escalation"}, "result": result})
        if result.get("created"):
            st.tickets.append(result)
        return result

    # -- main entry -----------------------------------------------------------
    def respond(self, message: str) -> dict:
        st = self.state
        calls: list[dict] = []
        message = (message or "").strip()
        self._handle_order(message, calls)

        products = catalog.find_products(message)
        if products:
            st.products = (products + [p for p in st.products if p not in products])[:4]
        queries, lang_hint = self._queries(message)
        hits = self.retriever.search(queries, k=self.k)
        extra_hits: list = []

        messages = [{"role": "system", "content": self.system}, *st.history[-12:],
                    {"role": "user", "content": self._context(message, hits, st.products)}]
        tickets_before = len(st.tickets)
        final = ""
        for _ in range(8):
            reply = self.llm.chat(messages, tools=TOOLS)
            tool_calls = getattr(reply, "tool_calls", None) or []
            if not tool_calls:
                final = reply.content or ""
                break
            messages.append({"role": "assistant", "content": reply.content or "",
                             "tool_calls": [{"id": tc.id, "type": "function", "function": {
                                 "name": tc.function.name, "arguments": tc.function.arguments},
                                 # Gemini 3 needs its thought_signature echoed back with each call
                                 **({"extra_content": tc.model_extra["extra_content"]}
                                    if (getattr(tc, "model_extra", None) or {}).get("extra_content") else {})}
                                 for tc in tool_calls]})
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self._run_tool(tc.function.name, args, calls, extra_hits)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, ensure_ascii=False, default=str)})
        data = parse_json(final)
        text = (data.get("reply") or final or "").strip()
        decision = str(data.get("decision", "answer")).lower()
        decision = decision if decision in DECISIONS else "answer"

        # Guard: never claim a handover that did not happen (ESC-2, ESC-4).
        created_now = len(st.tickets) > tickets_before
        if decision == "escalate" and not created_now:
            result = self._fallback_ticket(message, calls)
            if not result.get("created"):
                text = ("I'm sorry, I couldn't raise a ticket for a colleague right now. "
                        "Please try again in a few minutes.")
        if not text:
            text = "Sorry, I didn't catch that. Could you say it again?"
            decision = "clarify"
        if st.turns == 0 and not re.search(r"\bAI\b|एआई|artificial intelligence", text, re.I):
            intro = ("मैं ZentroMart की AI सहायक हूँ। " if has_devanagari(message)
                     else "I'm ZentroMart's AI support assistant. ")
            text = intro + text
        text = orders.scrub(text, st.secrets)

        sources = _valid_sources(data.get("sources"))
        language = data.get("language") or lang_hint or ("hi" if has_devanagari(text) else "en")
        st.history += [{"role": "user", "content": message}, {"role": "assistant", "content": text}]
        st.turns += 1
        retrieved = [h.as_dict() for h in hits + extra_hits]
        always_on = [sid for sid in chunk_by_id() if sid.split("-")[0] in ALWAYS_ON]
        return {"reply": text, "decision": decision, "sources": sources,
                "retrieved": retrieved, "retrieved_ids": [r["section_id"] for r in retrieved],
                "context_ids": always_on + [r["section_id"] for r in retrieved],
                "catalog_rows": [f"products.csv:{p['sku']}" for p in products],
                "tool_calls": calls, "language": language, "queries": queries,
                "order_state": st.describe()}
