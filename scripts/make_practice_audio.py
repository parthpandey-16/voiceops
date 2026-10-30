#!/usr/bin/env python
"""Turn the practice set into test audio with Gnani TTS, so run.py can be tested end to end.

    python scripts/make_practice_audio.py --out test_audio --questions 10 --conversations 4

Writes test_audio/questions/PQxxxx.wav (one call each) and
test_audio/conversations/PCxxx_turnN.wav (one call per conversation ID).
This is synthetic speech from Gnani only; record your own voice for the demo.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voiceops.config import settings  # noqa: E402
from voiceops.textnorm import has_devanagari  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_audio")
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--conversations", type=int, default=4)
    args = parser.parse_args()
    from gnani.tts import GnaniTTSClient

    tts = GnaniTTSClient()
    out = Path(args.out)
    practice = settings.dataset_dir / "practice"

    def say(text: str, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        voice, lang = ("Nalini", "hi-IN") if has_devanagari(text) else ("Pranav", "en-IN")
        tts.synthesize(text, voice=voice, model=settings.tts_model, language=lang, output_file=str(path))
        print("wrote", path)

    questions = list(csv.DictReader((practice / "questions.csv").open(encoding="utf-8")))
    for q in questions[: args.questions]:
        say(q["question"], out / "questions" / f"{q['id']}.wav")
    convs: dict[str, list[dict]] = {}
    for row in csv.DictReader((practice / "conversations.csv").open(encoding="utf-8")):
        convs.setdefault(row["conversation_id"], []).append(row)
    for cid in sorted(convs)[: args.conversations]:
        for row in convs[cid]:
            say(row["customer_message"], out / "conversations" / f"{cid}_turn{row['turn']}.wav")


if __name__ == "__main__":
    main()
