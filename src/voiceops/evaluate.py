"""Evaluate the agent on the practice set, the same way for every step.

    python -m voiceops.evaluate --step all          # Step 1, Step 2 (+ASR text), Step 3
    python -m voiceops.evaluate --step 2 --limit 10

Scores follow the dataset's own rules (DATA_DICTIONARY.md): every must_include
fact is required, " / " alternatives are equivalent, dates may be written any
common way, must_not_include phrases fail the reply, and a single question
passes if the decision is in acceptable_decisions. Results are resumable: rows
already in results/*.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

from .config import ROOT, settings
from .textnorm import MONTHS, ascii_digits

RESULTS = ROOT / "results"
_HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर",
              "अक्टूबर", "नवंबर", "दिसंबर"]
_NUM_WORDS = {
    1: ["one", "ek", "एक"], 2: ["two", "दो"], 3: ["three", "teen", "तीन"],
    4: ["four", "char", "चार"], 5: ["five", "paanch", "पांच", "पाँच"], 6: ["six", "chhe", "छह"],
    7: ["seven", "saat", "सात"], 8: ["eight", "aath", "आठ"], 9: ["nine", "nau", "नौ"],
    10: ["ten", "das", "दस"], 12: ["twelve", "barah", "बारह"], 15: ["fifteen", "pandrah", "पंद्रह"],
    30: ["thirty", "tees", "तीस"], 48: ["forty-eight", "forty eight", "अड़तालीस"],
    49: ["forty-nine", "forty nine", "उनचास"], 79: ["seventy-nine", "seventy nine"],
}


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", ascii_digits(text or "")).lower()
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # 10,000 -> 10000
    text = text.replace("’", "'").replace("‘", "'")
    # nukta-insensitive Devanagari (आख़िरी == आखिरी)
    return text.replace("़", "")


def _date_variants(d: date) -> list[str]:
    m, hm = MONTHS[d.month - 1].lower(), _HI_MONTHS[d.month - 1]
    day = d.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return [d.isoformat(), f"{day} {m}", f"{day}{suffix} {m}", f"{m} {day}", f"{day}{suffix} of {m}",
            f"{day} {m[:3]}", f"{m[:3]} {day}", f"{day}/{d.month}/{d.year}", f"{day:02d}/{d.month:02d}/{d.year}",
            norm(f"{day} {hm}")]


def _parse_date(alt: str) -> date | None:
    alt = alt.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", alt):
        return date.fromisoformat(alt)
    match = re.fullmatch(r"(\d{1,2}) ([A-Za-z]+) (\d{4})", alt)
    if match and match.group(2).capitalize() in MONTHS:
        return date(int(match.group(3)), MONTHS.index(match.group(2).capitalize()) + 1, int(match.group(1)))
    return None


def alt_matches(alt: str, reply_n: str) -> bool:
    alt = alt.strip()
    if not alt:
        return True
    d = _parse_date(alt)
    if d:
        return any(v in reply_n for v in _date_variants(d))
    alt_n = norm(alt)
    if re.fullmatch(r"\d+", alt_n):
        if re.search(rf"(?<![\d.]){alt_n}(?![\d])", reply_n):
            return True
        return any(re.search(rf"\b{re.escape(w)}\b", reply_n) if w.isascii() else w in reply_n
                   for w in _NUM_WORDS.get(int(alt_n), []))
    return alt_n in reply_n


def facts_ok(must_include: str, reply: str) -> tuple[bool, list[str]]:
    reply_n = norm(reply)
    missing = []
    for fact in [f for f in (must_include or "").split(" ; ") if f.strip()]:
        if not any(alt_matches(a, reply_n) for a in fact.split(" / ")):
            missing.append(fact)
    return not missing, missing


def forbidden_hits(must_not: str, reply: str) -> list[str]:
    reply_n = norm(reply)
    return [p for p in (must_not or "").split(" ; ") if p.strip() and norm(p.strip()) in reply_n]


def split_ids(value: str) -> list[str]:
    return [v.strip() for v in (value or "").split(" ; ") if v.strip()]


def retrieval_scores(gold: list[str], retrieved: list[str], cited: list[str], catalog: list[str],
                     ranked: list[str] | None = None) -> dict:
    """retrieved = every section in the model's context; ranked = the search ranking alone."""
    if not gold:
        return {}
    found = set(retrieved) | set(catalog)
    used = set(cited) | set(catalog)
    return {
        "recall_retrieved": len(set(gold) & found) / len(gold),
        "recall_cited": len(set(gold) & used) / len(gold),
        "precision_cited": (len(set(gold) & set(cited)) / len(set(cited))) if cited else 0.0,
        "hit_retrieved": float(bool(set(gold) & found)),
        "mrr": next((1 / (i + 1) for i, r in enumerate(ranked or retrieved) if r in gold), 0.0),
    }


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _csv(name: str) -> list[dict]:
    with open(settings.dataset_dir / "practice" / name, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["key"]: r for r in rows if not r.get("error")}


def _append(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def score_question(q: dict, out: dict) -> dict:
    acceptable = split_ids(q["acceptable_decisions"]) or [q["expected_decision"]]
    ok_facts, missing = facts_ok(q["must_include"], out["reply"])
    bad = forbidden_hits(q["must_not_include"], out["reply"])
    decision_ok = out["decision"] in acceptable
    gold = split_ids(q["gold_evidence"])
    return {"decision_ok": decision_ok, "facts_ok": ok_facts, "missing": missing, "forbidden": bad,
            "correct": decision_ok and ok_facts and not bad,
            "retrieval": retrieval_scores(gold, out.get("context_ids") or out.get("retrieved_ids", []),
                                          out.get("sources", []), out.get("catalog_rows", []),
                                          out.get("retrieved_ids", []))}


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_questions(step: int, use_asr: bool = False, limit: int | None = None, tag: str = "",
                  agent_kwargs: dict | None = None) -> Path:
    from .agent import BaselineAgent, SupportAgent
    from .llm import LLM

    llm = LLM()
    name = f"step{step}_questions{'_asr' if use_asr else ''}{tag}.jsonl"
    path = RESULTS / name
    done = _done(path)
    if step == 1:
        agent = BaselineAgent(llm)
    else:
        agent = SupportAgent(llm, **(agent_kwargs or {}))
    questions = _csv("questions.csv")[:limit] if limit else _csv("questions.csv")
    for i, q in enumerate(questions, 1):
        if q["id"] in done:
            continue
        text = q["asr_text"] if use_asr else q["question"]
        started = time.time()
        try:
            if step == 1:
                out = agent.answer(text)
            else:
                agent.reset()
                out = agent.respond(text)
            row = {"key": q["id"], "id": q["id"], "language": q["language"], "intent": q["intent"],
                   "input": text, **{k: out.get(k) for k in (
                       "reply", "decision", "sources", "retrieved_ids", "context_ids", "catalog_rows",
                       "tool_calls",
                       "queries")},
                   "expected_decision": q["expected_decision"], "gold_evidence": q["gold_evidence"],
                   "seconds": round(time.time() - started, 2)}
            row["score"] = score_question(q, out)
        except Exception as exc:  # noqa: BLE001
            logging.exception("question %s failed", q["id"])
            row = {"key": q["id"], "id": q["id"], "error": str(exc)}
        _append(path, row)
        mark = "ok " if row.get("score", {}).get("correct") else "MISS"
        print(f"[step{step}{'-asr' if use_asr else ''}{tag} {i}/{len(questions)}] {mark} {q['id']} "
              f"{row.get('decision', '')} {row.get('sources', '')}")
    return path


def _expected_calls(spec: str) -> list[dict]:
    calls = []
    for match in re.finditer(r"(\w+)\(([^)]*)\)", spec or ""):
        args = dict(re.findall(r"(\w+)=([^,\s)]+)", match.group(2)))
        calls.append({"tool": match.group(1), "args": args})
    return calls


def tool_calls_ok(expected: list[dict], actual: list[dict]) -> bool:
    actual_lookups = [c for c in actual if c["tool"] == "lookup_order"]
    for exp in expected:
        if exp["tool"] == "lookup_order":
            wants_proof = "verification_value" in exp["args"]
            if not any(c["args"].get("order_id") == exp["args"].get("order_id")
                       and (("verification_value" in c["args"]) == wants_proof) for c in actual_lookups):
                return False
        elif exp["tool"] == "create_ticket":
            if not any(c["tool"] == "create_ticket" and c["result"].get("created") for c in actual):
                return False
    # A ticket nobody asked for is a wrong handover.
    if not any(e["tool"] == "create_ticket" for e in expected) and any(
            c["tool"] == "create_ticket" for c in actual):
        return False
    return True


def run_conversations(limit: int | None = None, tag: str = "", agent_kwargs: dict | None = None) -> Path:
    from .agent import SupportAgent
    from .llm import LLM

    agent = SupportAgent(LLM(), **(agent_kwargs or {}))
    path = RESULTS / f"step3_conversations{tag}.jsonl"
    done = _done(path)
    convs: dict[str, list[dict]] = defaultdict(list)
    for row in _csv("conversations.csv"):
        convs[row["conversation_id"]].append(row)
    ids = sorted(convs)[:limit] if limit else sorted(convs)
    for n, cid in enumerate(ids, 1):
        turns = sorted(convs[cid], key=lambda r: int(r["turn"]))
        if all(f"{cid}:{t['turn']}" in done for t in turns):
            continue
        agent.reset()
        for t in turns:
            key = f"{cid}:{t['turn']}"
            try:
                out = agent.respond(t["customer_message"])
                ok_facts, missing = facts_ok(t["must_include"], out["reply"])
                bad = forbidden_hits(t["must_not_include"], out["reply"])
                tools_ok = tool_calls_ok(_expected_calls(t["expected_tool_calls"]), out["tool_calls"])
                decision_ok = out["decision"] == t["expected_decision"]
                row = {"key": key, "conversation_id": cid, "turn": int(t["turn"]), "scenario": t["scenario"],
                       "language": t["language"], "input": t["customer_message"], "reply": out["reply"],
                       "decision": out["decision"], "expected_decision": t["expected_decision"],
                       "sources": out["sources"], "retrieved_ids": out["retrieved_ids"],
                       "tool_calls": out["tool_calls"], "expected_tool_calls": t["expected_tool_calls"],
                       "gold_evidence": t["gold_evidence"],
                       "score": {"decision_ok": decision_ok, "facts_ok": ok_facts, "missing": missing,
                                 "forbidden": bad, "tools_ok": tools_ok,
                                 "correct": decision_ok and ok_facts and not bad and tools_ok,
                                 "retrieval": retrieval_scores(split_ids(t["gold_evidence"]),
                                                               out.get("context_ids") or out["retrieved_ids"],
                                                               out["sources"],
                                                               out.get("catalog_rows", []),
                                                               out["retrieved_ids"])}}
            except Exception as exc:  # noqa: BLE001
                logging.exception("conversation %s failed", key)
                row = {"key": key, "conversation_id": cid, "error": str(exc)}
            _append(path, row)
        print(f"[step3{tag} {n}/{len(ids)}] {cid} done")
    return path


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _mean(values):
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def summarise(path: Path) -> dict:
    rows = [r for r in _done(path).values()]
    if not rows:
        return {}
    scores = [r["score"] for r in rows]
    retr = [s["retrieval"] for s in scores if s.get("retrieval")]
    out = {"n": len(rows),
           "accuracy": _mean(s["correct"] for s in scores),
           "decision_accuracy": _mean(s["decision_ok"] for s in scores),
           "facts_accuracy": _mean(s["facts_ok"] for s in scores),
           "forbidden_rate": _mean(bool(s["forbidden"]) for s in scores)}
    if any("tools_ok" in s for s in scores):
        out["tool_call_accuracy"] = _mean(s["tools_ok"] for s in scores)
    if retr:
        for key in ("recall_retrieved", "recall_cited", "precision_cited", "hit_retrieved", "mrr"):
            out[key] = _mean(r[key] for r in retr)
    group_key = "intent" if "intent" in rows[0] else "scenario"
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(group_key, "?")].append(r["score"]["correct"])
    out[f"by_{group_key}"] = {k: {"n": len(v), "accuracy": _mean(v)} for k, v in sorted(groups.items())}
    langs = defaultdict(list)
    for r in rows:
        langs[r.get("language", "?")].append(r["score"]["correct"])
    out["by_language"] = {k: {"n": len(v), "accuracy": _mean(v)} for k, v in sorted(langs.items())}
    return out


def write_metrics() -> dict:
    metrics = {p.stem: summarise(p) for p in sorted(RESULTS.glob("*.jsonl"))}
    metrics["_meta"] = {"provider": settings.provider, "model": settings.llm_model,
                        "embeddings": settings.embeddings, "top_k": settings.top_k}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def retrieval_ablation(k: int | None = None) -> dict:
    """Retrieval-only recall@k on the 52 questions for each retrieval component.

    Needs no LLM calls: the rewritten queries are read back from the Step 2 run.
    """
    from .retriever import Retriever

    k = k or settings.top_k
    questions = [q for q in _csv("questions.csv")
                 if [g for g in split_ids(q["gold_evidence"]) if not g.startswith("products.csv")]]
    rewrites = {r["id"]: r.get("queries") or [] for r in _done(RESULTS / "step2_questions.jsonl").values()}
    has_expansion = any(settings.cache_dir.glob("expansions_*.json"))
    configs = [("BM25 only", dict(embeddings="none", expand=False), False)]
    configs.append(("BM25 + dense", dict(embeddings=None, expand=False), False))
    if has_expansion:
        configs.append(("BM25 + dense + doc expansion", dict(embeddings=None, expand=True), False))
    if rewrites:
        configs.append(("full: + LLM query rewrite", dict(embeddings=None, expand=has_expansion), True))
    out = {}
    for name, kwargs, use_rewrite in configs:
        retriever = Retriever(llm=None, **kwargs)
        if name != "BM25 only" and retriever.matrix is None:
            name += " (dense unavailable)"
        recalls, hits = [], []
        for q in questions:
            gold = {g for g in split_ids(q["gold_evidence"]) if not g.startswith("products.csv")}
            gold -= {sid for sid in gold if sid.split("-")[0] in ("STY", "IDV", "ESC", "KB")}
            if not gold:
                continue
            queries = rewrites.get(q["id"], []) if use_rewrite else [q["question"]]
            ids = [h.chunk.section_id for h in retriever.search(queries or [q["question"]], k=k)]
            recalls.append(len(gold & set(ids)) / len(gold))
            hits.append(float(bool(gold & set(ids))))
        out[name] = {"recall_at_k": _mean(recalls), "hit_at_k": _mean(hits), "n": len(recalls), "k": k}
        print(f"{name:40s} recall@{k}={out[name]['recall_at_k']}  hit@{k}={out[name]['hit_at_k']}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval_ablation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", default="all", choices=["1", "2", "2asr", "3", "ablation", "retrieval", "all"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh", action="store_true", help="delete earlier results first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.fresh:
        for p in RESULTS.glob("*.jsonl"):
            p.unlink()
    if args.step in ("1", "all"):
        run_questions(1, limit=args.limit)
    if args.step in ("2", "all"):
        run_questions(2, limit=args.limit)
    if args.step in ("2asr", "all"):
        run_questions(2, use_asr=True, limit=args.limit)
    if args.step in ("3", "all"):
        run_conversations(limit=args.limit)
    if args.step == "ablation":
        # What each retrieval component buys us (reported in the write-up).
        run_questions(2, limit=args.limit, tag="_no_rewrite", agent_kwargs={"rewrite": False})
    if args.step in ("retrieval", "ablation", "all"):
        retrieval_ablation()
    metrics = write_metrics()
    for name, m in metrics.items():
        if name.startswith("_") or not m:
            continue
        keys = ("accuracy", "decision_accuracy", "facts_accuracy", "recall_retrieved", "recall_cited",
                "tool_call_accuracy")
        print(f"{name:32s} " + "  ".join(f"{k}={m[k]}" for k in keys if k in m))


if __name__ == "__main__":
    main()
