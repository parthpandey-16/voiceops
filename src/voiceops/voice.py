"""Step 4: Gnani STT in, Gnani TTS out. The agent's brain is unchanged.

Only Gnani's speech APIs are used, as the capstone rules require. Fallbacks
stay inside Gnani (another Gnani voice or language), never another provider.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from pathlib import Path

from .config import settings
from .textnorm import for_speech, has_devanagari

log = logging.getLogger(__name__)

NOT_HEARD = {
    "en": "Sorry, I didn't catch that. Could you please say it again?",
    "hi": "माफ़ कीजिए, मैं सुन नहीं पाई। क्या आप दोबारा बोल सकते हैं?",
    "hi-en": "Sorry, main sun nahi paayi. Kya aap dobara bol sakte hain?",
}
STILL_NOT_HEARD = {
    "en": "I'm still having trouble hearing you. Please try again in a quieter spot, "
          "or I can connect you to a human colleague.",
    "hi": "मुझे अब भी ठीक से सुनाई नहीं दे रहा। कृपया शांत जगह से दोबारा बोलें, "
          "या मैं आपको किसी सहयोगी से जोड़ सकती हूँ।",
    "hi-en": "Mujhe abhi bhi theek se sunai nahi de raha. Thodi shaant jagah se dobara boliye, "
             "ya main aapko colleague se connect kar sakti hoon.",
}
_FILLERS = {"uh", "um", "umm", "hmm", "hm", "ah", "er", "mm", "uhh"}
MAX_CLIP_SECONDS = 55  # the STT REST endpoint accepts clips up to 60 s


def _lang_family(text: str, hint: str = "") -> str:
    if has_devanagari(text) or hint == "hi":
        return "hi"
    if hint in ("hi-en", "hinglish"):
        return "hi-en"
    return "en"


def is_unclear(transcript: str) -> bool:
    words = re.findall(r"[\wऀ-ॿ]+", transcript.lower())
    return not words or all(w in _FILLERS for w in words)


class Voice:
    def __init__(self):
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient

        if not os.getenv("GNANI_API_KEY"):
            raise RuntimeError("Set GNANI_API_KEY (see .env.example).")
        self.stt = GnaniSTTClient()
        self.tts = GnaniTTSClient()

    # -- ears ---------------------------------------------------------------
    def _split(self, path: Path) -> list[Path]:
        """Cut clips longer than the REST limit into pieces (WAV/FLAC/OGG via soundfile)."""
        try:
            import soundfile as sf

            info = sf.info(str(path))
        except Exception:  # noqa: BLE001 - mp3/m4a etc.: send as-is
            return [path]
        if info.duration <= MAX_CLIP_SECONDS:
            return [path]
        data, rate = sf.read(str(path))
        step = int(MAX_CLIP_SECONDS * rate)
        out = []
        tmp = Path(tempfile.mkdtemp(prefix="voiceops_"))
        for i in range(0, len(data), step):
            piece = tmp / f"{path.stem}_{i // step}.wav"
            sf.write(str(piece), data[i:i + step], rate)
            out.append(piece)
        return out

    def transcribe(self, audio_path: str | Path, language_code: str | None = None) -> str:
        language_code = language_code or settings.stt_language
        # ITN ("transcribe" format) writes numbers as digits, which keeps order
        # IDs and phone digits intact. It is offered for en-IN and hi-IN.
        fmt = "transcribe" if language_code in ("en-IN", "hi-IN") else "verbatim"
        texts = []
        for piece in self._split(Path(audio_path)):
            for attempt in range(3):
                try:
                    result = self.stt.transcribe(str(piece), language_code=language_code, format=fmt)
                    texts.append((result.get("transcript") or "").strip())
                    break
                except Exception as exc:  # noqa: BLE001
                    log.warning("STT failed on %s (%s), attempt %d", piece.name, exc, attempt + 1)
                    time.sleep(2 * (attempt + 1))
        return " ".join(t for t in texts if t).strip()

    # -- voice --------------------------------------------------------------
    def _voice_for(self, family: str) -> list[tuple[str, str]]:
        if family == "hi":
            return [(settings.tts_voice_hi, "hi-IN"), (settings.tts_voice, "en-IN")]
        if family == "hi-en":
            return [(settings.tts_voice_hinglish, "hi-IN"), (settings.tts_voice, "en-IN")]
        return [(settings.tts_voice, "en-IN")]

    def synthesize(self, text: str, output_file: str | Path, language_hint: str = "") -> str:
        family = _lang_family(text, language_hint)
        spoken = for_speech(text, devanagari=(family == "hi"))
        last_exc = None
        for voice, language in self._voice_for(family):
            try:
                self.tts.synthesize(spoken, voice=voice, model=settings.tts_model,
                                    language=language, output_file=str(output_file))
                return spoken
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("TTS failed with %s/%s: %s", voice, language, exc)
        raise RuntimeError(f"Gnani TTS failed: {last_exc}")


class VoiceSession:
    """One phone call: audio in -> STT -> agent -> TTS -> audio out."""

    def __init__(self, agent, voice: Voice):
        self.agent, self.voice = agent, voice
        self.misses = 0
        self.language = "en"

    def turn(self, audio_path: str | Path, reply_wav: str | Path) -> dict:
        heard = self.voice.transcribe(audio_path)
        if is_unclear(heard):
            self.misses += 1
            table = STILL_NOT_HEARD if self.misses >= 2 else NOT_HEARD
            result = {"reply": table.get(self.language, table["en"]), "decision": "clarify",
                      "sources": [], "retrieved": [], "retrieved_ids": [], "tool_calls": [],
                      "language": self.language, "unclear_audio": True}
        else:
            self.misses = 0
            result = self.agent.respond(heard)
            self.language = _lang_family(result["reply"], result.get("language", ""))
        result["transcript"] = heard
        result["speech_text"] = self.voice.synthesize(result["reply"], reply_wav,
                                                      result.get("language", ""))
        return result
