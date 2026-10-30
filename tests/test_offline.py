"""Offline tests: no API keys needed. Run with `python -m pytest tests -q`."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from voiceops import dates  # noqa: E402
from voiceops.agent import SupportAgent  # noqa: E402
from voiceops.catalog import find_products  # noqa: E402
from voiceops.evaluate import _csv, _expected_calls, facts_ok, forbidden_hits, tool_calls_ok  # noqa: E402
from voiceops.retriever import Retriever  # noqa: E402
from voiceops.textnorm import (find_order_id, find_skus, find_verification_value,  # noqa: E402
                               for_speech)


class ScriptedLLM:
    """Plays back canned model turns; each item is a final JSON dict or a list of tool calls."""

    can_embed = False

    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def complete(self, prompt, system=None, json_mode=False):
        return json.dumps({"query": prompt[-200:], "language": "en"})

    def chat(self, messages, tools=None, **_):
        self.seen.append(messages)
        step = self.script.pop(0)
        if isinstance(step, list):
            calls = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(
                name=name, arguments=json.dumps(args))) for i, (name, args) in enumerate(step)]
            return SimpleNamespace(content="", tool_calls=calls)
        return SimpleNamespace(content=json.dumps(step), tool_calls=None)


def test_business_days_match_practice_answers():
    assert dates.add_business_days("2026-10-17", 7).isoformat() == "2026-10-27"  # skips Sun + 19 Oct
    assert dates.add_business_days("2026-10-12", 3).isoformat() == "2026-10-15"
    assert dates.add_business_days("2026-10-14", 5).isoformat() == "2026-10-21"


def test_identifier_extraction():
    assert find_order_id("Order number hai 8978 2837.") == "89782837"
    assert find_order_id("order eight nine seven eight two eight three seven") == "89782837"
    assert find_verification_value("आख़िरी चार अंक 0783 हैं।") == "0783"
    assert find_verification_value("Pincode 440010 hai.") == "440010"
    assert find_verification_value("refund 17 October 2026 ko shuru hua") is None
    assert find_skus("so sku ele-zero zero one to pe zentromart assured") == ["ELE-0012"]


def test_catalog_matching():
    assert [p["sku"] for p in find_products("Pagecraft A5 Ruled Notebook (Pack of 4)")] == ["BKS-0056"]
    assert [p["sku"] for p in find_products("Orbix Fit 3 Pro Smartwatch")] == ["ELE-0017"]


def test_speech_formatting():
    spoken = for_speech("Order 89782837 arrives on 2026-10-27 and costs ₹49 [RET-6].")
    assert "8978 2837" in spoken and "27th of October 2026" in spoken and "49 rupees" in spoken
    assert "[" not in spoken


def test_scorer_accepts_reference_answers():
    for q in _csv("questions.csv"):
        ok, missing = facts_ok(q["must_include"], q["reference_answer"])
        assert ok, (q["id"], missing)
        assert not forbidden_hits(q["must_not_include"], q["reference_answer"])


def test_bm25_retrieval_finds_core_sections():
    r = Retriever(llm=None, embeddings="none", expand=False)
    ids = [h.chunk.section_id for h in r.search("How long does a refund to UPI take?")]
    assert "REF-2" in ids


def test_conversation_flow_with_verification_and_ticket():
    retriever = Retriever(llm=None, embeddings="none", expand=False)
    llm = ScriptedLLM([
        {"reply": "Hi, I'm ZentroMart's AI assistant. Order ID batayein?", "decision": "clarify",
         "sources": ["STY-1"]},
        {"reply": "Phone ke last 4 digits ya pincode batayein.", "decision": "clarify",
         "sources": ["IDV-2"]},
        {"reply": "Shukriya, verify ho gaya.", "decision": "answer", "sources": ["IDV-2"]},
        [("create_ticket", {"summary": "Verified caller reports a damaged Zentro Vista TV "
                                       "worth over 10000; Priority Resolution must decide.",
                            "reason": "high_value_damage"})],
        {"reply": "Main approve nahi kar sakti; Priority Resolution team aapse sampark karegi.",
         "decision": "escalate", "sources": ["DWM-3", "ESC-1"]},
    ])
    agent = SupportAgent(llm, retriever=retriever)
    turns = [t for t in _csv("conversations.csv") if t["conversation_id"] == "PC002"]
    for turn in turns:
        out = agent.respond(turn["customer_message"])
        assert tool_calls_ok(_expected_calls(turn["expected_tool_calls"]), out["tool_calls"]), turn["turn"]
        assert out["decision"] == turn["expected_decision"]
    # verified order data reached the model only after verification
    assert "VERIFIED ORDER" not in llm.seen[1][-1]["content"]
    assert "VERIFIED ORDER" in llm.seen[2][-1]["content"]
    assert "4533" not in json.dumps(out["tool_calls"])


def test_failed_verification_locks_after_retry():
    retriever = Retriever(llm=None, embeddings="none", expand=False)
    llm = ScriptedLLM([{"reply": "AI assistant here.", "decision": "clarify"}] * 4)
    agent = SupportAgent(llm, retriever=retriever)
    for msg in ["नमस्ते", "ऑर्डर नंबर है 7746 1439।", "आख़िरी चार अंक 1266 हैं।", "आख़िरी चार अंक 1266 हैं।"]:
        agent.respond(msg)
    assert agent.state.locked and not agent.state.verified
    assert "FAILED twice" in llm.seen[-1][-1]["content"]


def test_escalation_without_ticket_is_repaired():
    retriever = Retriever(llm=None, embeddings="none", expand=False)
    llm = ScriptedLLM([{"reply": "A human colleague will contact you.", "decision": "escalate"}])
    agent = SupportAgent(llm, retriever=retriever)
    out = agent.respond("I want to talk to a human, there is an order I never placed")
    assert any(c["tool"] == "create_ticket" and c["result"]["created"] for c in out["tool_calls"])


def test_run_grouping(tmp_path):
    from run import group_files

    for name in ["PC002_turn1.wav", "PC002_turn2.wav", "PQ0003.wav"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "call7").mkdir()
    (tmp_path / "call7" / "a.wav").write_bytes(b"")
    groups = group_files(tmp_path, "auto")
    assert [p.name for p in groups["PC002"]] == ["PC002_turn1.wav", "PC002_turn2.wav"]
    assert "PQ0003" in groups and "call7" in groups
