#!/usr/bin/env python
"""Build report.pdf (4 pages max) from results/metrics.json and the result files.

    python scripts/build_report.py [--team "Team name"] [--repo https://github.com/...]

Numbers always come from the latest evaluation run, so re-run this after
`python evaluate.py --step all`. The narrative paragraphs live in NOTES below;
edit them to reflect what you saw in your own run.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

RESULTS = ROOT / "results"

NOTES = {
    "step1": "A single LLM call with a system prompt that sets the role (ZentroMart's AI support "
             "assistant), the limits (no cancelling, refunding or approving; never share other "
             "customers' data; say so when unsure) and the AI disclosure. It has no documents, so it "
             "falls back on generic e-commerce knowledge: it guesses plausible but wrong windows "
             "and timelines (for example 5 to 7 days for every refund), and it invents policies for "
             "questions the documents do not cover.",
    "step2": "Each of the 84 Markdown sections becomes one chunk carrying its section ID, doc_type, "
             "status and effective_from. Retrieval is hybrid: BM25 for exact terms and SKUs, plus "
             "multilingual dense embeddings (paraphrase-multilingual-MiniLM via fastembed) for "
             "paraphrases and Hindi questions, fused with reciprocal-rank fusion. An LLM rewrites each "
             "message into a standalone English query, and an LLM-generated set of customer-style "
             "questions per section (document expansion, cached) bridges the vocabulary gap "
             "(\"send it back\" vs. \"reverse pickup\"). Explicit cross-references (REF-1, DWM-1) "
             "are followed, and a light prior applies the KB-00.1 precedence. Catalog rows are matched "
             "by SKU (also spoken SKUs such as \"ele zero zero one two\") or by product name. The "
             "model answers only from what it is given, cites the section IDs it used, uses an exact "
             "date_calc tool for business-day maths, and replies \"I don't have that information\" "
             "with an offer of a colleague when nothing covers the question.",
    "step3": "A ConversationState keeps the order ID, verification status, items discussed, tickets "
             "and the last 12 messages, so follow-ups such as \"and the refund?\" resolve to the same "
             "order and item. The identity check is deterministic code, not left to the model: "
             "lookup_order(order_id) runs as soon as an ID is heard, and lookup_order(order_id, "
             "value) runs when a 4-digit phone suffix or 6-digit pincode arrives, with one retry and "
             "then a lock (IDV-3). The model only sees order data after verification, never sees the "
             "verification digits, and verification digits are scrubbed from replies and ticket "
             "summaries. Handover calls create_ticket with an ESC-3 summary, and a guard makes sure "
             "no reply claims a handover the tool did not confirm (created=True). There is no cancel, "
             "refund or credit tool, and the prompt forbids claiming those actions.",
    "step4": "Gnani STT (REST, format=\"transcribe\" so spoken numbers come back as digits) feeds the "
             "unchanged Step 3 agent, and Gnani TTS (timbre-v2.5: Kaveri for English, Nalini for "
             "Hindi, Poorvi for Hinglish) speaks the reply. Replies are rewritten for the ear: dates "
             "as \"the 27th of October 2026\", order IDs in groups of four, \"Rs 49\" as \"49 rupees\", "
             "with no markdown or section IDs. Empty or filler-only transcripts get a short \"Sorry, I "
             "didn't catch that\", and a second miss offers a human colleague. Clips over 55 s are "
             "split for the 60 s REST limit. run.py processes a folder of audio, keeps one memory per "
             "conversation and saves the transcript, text reply, WAV reply, decision, tool calls, "
             "cited sections and all retrieved chunks for each file.",
    "wrong": [
        "Documents disagree by design: FAQ-1 says refunds take 5 to 7 days, while REF-2 says UPI "
        "takes 1 to 3 business days and the Wallet is instant. A model without the documents gives "
        "the generic answer, and a weak retriever surfaces the FAQ. Fix: precedence (KB-00.1) is in the system prompt, and each chunk is labelled with "
        "its doc_type and status, so the policy wins.",
        "Date maths: LLMs miscount business days, and this dataset puts Sunday 18 October next to "
        "the 19 October warehouse holiday. Fix: an exact date_calc tool, plus a rule that a \"by\" date is the end "
        "of the range.",
        "Version traps: electronics delivered before 1 August 2026 keep the 10-day window from the "
        "superseded v3 policy, and festive-sale fashion gets 30 days. Superseded chunks are kept but "
        "labelled, and their scope note is copied onto every chunk.",
        "Hindi and Hinglish questions against English documents defeat BM25, and so does ASR text "
        "with spelled-out numbers. Fix: query rewrite into English, multilingual embeddings, and a "
        "spoken-digit normaliser for order IDs, SKUs and phone digits.",
        "A model can say \"I've raised a ticket\" without calling the tool. Fix: the decision is "
        "checked in code, and if a handover was declared without a successful create_ticket, the "
        "code creates the ticket or changes the reply.",
    ],
    "next": "Next steps: (1) a cross-encoder reranker to raise precision on the top three chunks; (2) "
            "streaming STT/TTS over Gnani's WebSocket API to cut latency on calls; (3) detecting the "
            "caller's language from a short first clip and switching STT between en-IN and hi-IN; "
            "(4) an LLM-as-judge pass for paraphrased must_not_include claims that string matching "
            "misses.",
    "bonus": "Bonus work: Hindi, Hinglish and English replies with language-matched Gnani voices; "
             "a browser demo (app.py) that shows retrieval, tool calls and identity-check state live; "
             "a resumable evaluation harness with per-intent and per-language breakdowns and a "
             "retrieval ablation; offline tests; and a script that turns the practice set into Gnani "
             "TTS audio for end-to-end tests of run.py.",
}


def pct(value) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def table(rows, widths, header=True):
    t = Table(rows, colWidths=widths)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def failures(name: str, limit: int = 6) -> list[str]:
    path = RESULTS / f"{name}.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if "score" in r and not r["score"]["correct"]]
    out = []
    for r in rows[:limit]:
        s = r["score"]
        why = []
        if not s["decision_ok"]:
            why.append(f"decision {r['decision']} (expected {r['expected_decision']})")
        if s["missing"]:
            # The PDF font has no Devanagari: keep the first Latin alternative of each fact
            latin = [next((a for a in m.split(" / ") if a.isascii()), m.encode("ascii", "ignore").decode())
                     for m in s["missing"]]
            why.append("missing " + ", ".join(latin)[:60])
        if s["forbidden"]:
            why.append("said " + ", ".join(s["forbidden"])[:60])
        if s.get("tools_ok") is False:
            why.append("tool calls")
        out.append(f"{r.get('id') or r.get('key')}: {'; '.join(why)}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="Team")
    parser.add_argument("--repo", default="https://github.com/<user>/<repo>")
    parser.add_argument("--out", default=str(ROOT / "report.pdf"))
    parser.add_argument("--llm-note", default=None, help="replaces the LLM model name, e.g. when runs used several models")
    args = parser.parse_args()

    metrics = json.loads((RESULTS / "metrics.json").read_text()) if (RESULTS / "metrics.json").exists() else {}
    ablation = json.loads((RESULTS / "retrieval_ablation.json").read_text()) \
        if (RESULTS / "retrieval_ablation.json").exists() else {}
    meta = metrics.get("_meta", {})
    s1, s2 = metrics.get("step1_questions", {}), metrics.get("step2_questions", {})
    s2a, s3 = metrics.get("step2_questions_asr", {}), metrics.get("step3_conversations", {})
    s2n = metrics.get("step2_questions_no_rewrite", {})

    styles = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=8.6, leading=11, spaceAfter=3)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5)
    P = lambda text, st=body: Paragraph(text, st)  # noqa: E731

    story = [P("ZentroMart VoiceOps agent: capstone report", h1),
             P(f"<b>{args.team}</b> · VoiceOps 2026, LLM &amp; Generative AI course (C&amp;A Club, IIT Guwahati × "
               f"Gnani.ai) · Code: {args.repo}", small),
             P(f"LLM: {args.llm_note or meta.get('model', 'n/a') + ' (' + meta.get('provider', 'n/a') + ')'} · embeddings: "
               f"{meta.get('embeddings', 'n/a')} · top-k {meta.get('top_k', 'n/a')} · speech: Gnani STT + TTS only · "
               "reference date 20 October 2026", small),
             Spacer(1, 4),
             P("Pipeline: customer audio → Gnani STT → agent (LLM + hybrid RAG + order/ticket tools + "
               "conversation memory) → Gnani TTS → spoken reply. The same agent object answers typed "
               "text, which is how Steps 1 to 3 are evaluated, and voice calls in Step 4.")]

    story += [P("1. What we built", h2)]
    for key, title in (("step1", "Step 1: prompt-only baseline."), ("step2", "Step 2: grounded RAG."),
                       ("step3", "Step 3: conversation."), ("step4", "Step 4: voice.")):
        story.append(P(f"<b>{title}</b> {NOTES[key]}"))

    story += [P("2. Step 1 vs Step 2 on the 52 practice questions", h2),
              P("Same scorer for both, following DATA_DICTIONARY.md: a reply is <b>correct</b> only if the "
                "decision is in acceptable_decisions, every must_include fact is present (alternatives and "
                "date formats are equivalent), and no must_not_include phrase appears. Retrieval recall counts "
                "gold section IDs and catalog rows found in the model's context; cited recall and precision "
                "use the section IDs the agent reported.")]
    rows = [["Metric", "Step 1 prompt-only", "Step 2 RAG", "Step 2 on ASR text", "Step 2 no rewrite"]]
    for label, key in (("Correct (all checks)", "accuracy"), ("Decision correct", "decision_accuracy"),
                       ("Required facts present", "facts_accuracy"), ("Forbidden claim made", "forbidden_rate"),
                       ("Gold evidence retrieved (recall)", "recall_retrieved"),
                       ("Gold evidence cited (recall)", "recall_cited"), ("Cited precision", "precision_cited")):
        rows.append([label] + [pct(m.get(key)) if m else "not run" for m in (s1, s2, s2a, s2n)])
    story.append(table(rows, [52 * mm, 30 * mm, 28 * mm, 32 * mm, 30 * mm]))
    if s1 and s2:
        story.append(P(f"Step 2 answers <b>{pct(s2.get('accuracy'))}</b> of the questions fully correctly, against "
                       f"<b>{pct(s1.get('accuracy'))}</b> for the prompt-only baseline."))

    intents = sorted(set(s1.get("by_intent", {})) | set(s2.get("by_intent", {})))
    if intents:
        rows = [["Intent", "n", "Step 1", "Step 2", "Intent", "n", "Step 1", "Step 2"]]
        cells = [[i, str(s2.get("by_intent", {}).get(i, s1.get("by_intent", {}).get(i, {})).get("n", "")),
                  pct(s1.get("by_intent", {}).get(i, {}).get("accuracy")),
                  pct(s2.get("by_intent", {}).get(i, {}).get("accuracy"))] for i in intents]
        half = (len(cells) + 1) // 2
        for a, b in zip(cells[:half], cells[half:] + [["", "", "", ""]]):
            rows.append(a + b)
        story += [Spacer(1, 4), table(rows, [34 * mm, 8 * mm, 15 * mm, 15 * mm] * 2)]
    if ablation:
        rows = [["Retrieval configuration (no LLM in the loop)", "recall@k", "hit@k"]]
        rows += [[name, pct(v["recall_at_k"]), pct(v["hit_at_k"])] for name, v in ablation.items()]
        story += [Spacer(1, 4), P("Retrieval ablation: each component's contribution, on gold sections other than "
                                  "the always-on agent guides.", small),
                  table(rows, [95 * mm, 25 * mm, 25 * mm])]

    story += [P("3. Step 3 on the 23 practice conversations (96 turns)", h2)]
    if s3:
        rows = [["Metric", "Value"], ["Turns fully correct", pct(s3.get("accuracy"))],
                ["Decision correct (answer / clarify / escalate / refuse)", pct(s3.get("decision_accuracy"))],
                ["Required facts present", pct(s3.get("facts_accuracy"))],
                ["Expected tool calls made, and no unrequested tickets", pct(s3.get("tool_call_accuracy"))],
                ["Forbidden claim made", pct(s3.get("forbidden_rate"))],
                ["Gold evidence cited (recall)", pct(s3.get("recall_cited"))]]
        story.append(table(rows, [110 * mm, 30 * mm]))
        sc = s3.get("by_scenario", {})
        if sc:
            rows = [["Scenario", "turns", "correct"]] + [[k, str(v["n"]), pct(v["accuracy"])] for k, v in sc.items()]
            story += [Spacer(1, 4), table(rows, [60 * mm, 18 * mm, 20 * mm])]
    else:
        story.append(P("Not run yet: run <font face='Courier'>python evaluate.py --step 3</font>."))

    story += [P("4. What went wrong, and what we changed", h2)]
    for item in NOTES["wrong"]:
        story.append(P("• " + item))
    fails = failures("step2_questions") + failures("step3_conversations")
    if fails:
        story.append(P("<b>Remaining misses in our run</b> (from results/*.jsonl): " + "; ".join(fails[:10]) + ".",
                       small))
    story.append(P(NOTES["next"]))
    story += [P("5. Voice and bonus work", h2), P(NOTES["bonus"]),
              P("Rules: speech goes through Gnani's API only, with no other STT or TTS anywhere, not even as a "
                "fallback. Only the synthetic dataset is used. Keys are read from the environment (.env.example "
                "has none). Every test answer is produced by the agent; nothing is hard-coded.", small)]

    doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=13 * mm, bottomMargin=13 * mm, title="VoiceOps 2026 report")
    doc.build(story)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
