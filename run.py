#!/usr/bin/env python
"""Batch run: a folder of customer audio in, spoken + written replies out.

    python run.py --input path/to/audio_folder --output sample_outputs/run1

For every audio file the agent saves:
    <output>/<conversation>/<file>.reply.wav    Gnani TTS reply
    <output>/<conversation>/<file>.reply.txt    reply text
    <output>/<conversation>/<file>.json         transcript, reply, decision, tool calls,
                                                cited sources and every retrieved chunk
plus <output>/results.csv and <output>/results.jsonl covering all files.

How files are grouped into calls (one agent memory per call):
    * a sub-folder is one conversation, its files in natural order;
    * flat files named <id>_turn<N> or <id>_t<N> form one conversation per <id>;
    * a manifest.csv (columns file, conversation_id, turn) overrides both;
    * any other file is its own single-turn call.
Use --mode single or --mode conversation to force one behaviour.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
_TURN_RE = re.compile(r"^(?P<conv>.+?)[_-](?:turn|t)(?P<turn>\d+)$", re.I)


def _natural(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def group_files(folder: Path, mode: str) -> "OrderedDict[str, list[Path]]":
    groups: OrderedDict[str, list[Path]] = OrderedDict()
    manifest = folder / "manifest.csv"
    if manifest.exists() and mode == "auto":
        rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
        rows.sort(key=lambda r: (r["conversation_id"], int(r.get("turn") or 0)))
        for row in rows:
            groups.setdefault(row["conversation_id"], []).append(folder / row["file"])
        return groups
    files = sorted((p for p in folder.rglob("*") if p.suffix.lower() in AUDIO), key=_natural)
    for path in files:
        rel = path.relative_to(folder)
        if mode == "single":
            key = str(rel.with_suffix(""))
        elif mode == "conversation":
            key = str(rel.parent) if rel.parent != Path(".") else "conversation"
        elif rel.parent != Path("."):
            key = str(rel.parent)
        else:
            match = _TURN_RE.match(path.stem)
            key = match.group("conv") if match else path.stem
        groups.setdefault(key, []).append(path)
    for key, paths in groups.items():
        def turn_no(p: Path):
            m = _TURN_RE.match(p.stem)
            return (int(m.group("turn")) if m else 0, _natural(p))
        paths.sort(key=turn_no)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", "-i", required=True, help="folder of customer audio files")
    parser.add_argument("--output", "-o", default="outputs/run", help="where replies are saved")
    parser.add_argument("--mode", choices=["auto", "single", "conversation"], default="auto")
    parser.add_argument("--stt-language", default=None, help="Gnani STT language, default en-IN")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    from voiceops.config import settings
    if args.stt_language:
        settings.stt_language = args.stt_language
    from voiceops.agent import SupportAgent
    from voiceops.llm import LLM
    from voiceops.voice import Voice, VoiceSession

    folder, out = Path(args.input), Path(args.output)
    if not folder.is_dir():
        parser.error(f"{folder} is not a folder")
    groups = group_files(folder, args.mode)
    if not groups:
        parser.error(f"no audio files ({', '.join(sorted(AUDIO))}) in {folder}")
    out.mkdir(parents=True, exist_ok=True)

    llm = LLM()
    agent = SupportAgent(llm, voice=True)
    voice = Voice()
    rows = []
    jsonl = (out / "results.jsonl").open("w", encoding="utf-8")
    total = sum(len(v) for v in groups.values())
    done = 0
    for conv, paths in groups.items():
        agent.reset()
        session = VoiceSession(agent, voice)
        conv_dir = out / re.sub(r"[^\w.-]+", "_", conv)
        conv_dir.mkdir(parents=True, exist_ok=True)
        for turn, path in enumerate(paths, 1):
            done += 1
            started = time.time()
            wav = conv_dir / f"{path.stem}.reply.wav"
            try:
                result = session.turn(path, wav)
                error = ""
            except Exception as exc:  # noqa: BLE001 - keep going; record the failure
                logging.exception("failed on %s", path)
                result, error = {"reply": "", "transcript": "", "decision": "", "sources": [],
                                 "retrieved": [], "tool_calls": []}, str(exc)
            record = {"file": str(path.relative_to(folder)), "conversation": conv, "turn": turn,
                      "transcript": result.get("transcript", ""), "reply": result.get("reply", ""),
                      "speech_text": result.get("speech_text", ""), "decision": result.get("decision", ""),
                      "sources": result.get("sources", []),
                      "retrieved_chunks": result.get("retrieved", []),
                      "tool_calls": result.get("tool_calls", []),
                      "reply_audio": str(wav.relative_to(out)) if wav.exists() else "",
                      "seconds": round(time.time() - started, 2), "error": error}
            (conv_dir / f"{path.stem}.reply.txt").write_text(record["reply"], encoding="utf-8")
            (conv_dir / f"{path.stem}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            jsonl.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            jsonl.flush()
            rows.append(record)
            print(f"[{done}/{total}] {record['file']}: heard {record['transcript'][:60]!r} -> "
                  f"{record['decision']} {record['sources']}")
    jsonl.close()
    with (out / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "conversation", "turn", "transcript", "reply",
                                                "decision", "sources", "retrieved_ids", "reply_audio",
                                                "seconds", "error"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in ("file", "conversation", "turn", "transcript", "reply",
                                                  "decision", "reply_audio", "seconds", "error")},
                             "sources": " ; ".join(r["sources"]),
                             "retrieved_ids": " ; ".join(c["section_id"] for c in r["retrieved_chunks"])})
    print(f"Saved {len(rows)} replies to {out}/ ({sum(1 for r in rows if r['error'])} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
