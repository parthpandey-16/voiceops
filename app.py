#!/usr/bin/env python
"""Live demo: talk to the ZentroMart agent in the browser.

    python app.py            # then open http://127.0.0.1:7860

Record with the microphone (Gnani STT -> agent -> Gnani TTS) or type a message.
The right-hand panel shows what the agent retrieved, which tools it called and
the identity-check state, which is what the demo video should show.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr  # noqa: E402

from voiceops.agent import SupportAgent  # noqa: E402
from voiceops.llm import LLM  # noqa: E402

AGENT = SupportAgent(LLM(), voice=True)
try:
    from voiceops.voice import Voice, VoiceSession

    VOICE = Voice()
except Exception as exc:  # noqa: BLE001 - text-only demo still works
    print(f"Voice disabled: {exc}")
    VOICE = None
SESSION = VoiceSession(AGENT, VOICE) if VOICE else None
OUT = Path(tempfile.mkdtemp(prefix="voiceops_demo_"))


def _panel(result: dict) -> str:
    tools = [{"tool": c["tool"], "args": c["args"],
              "result": {k: v for k, v in c["result"].items() if k in (
                  "created", "ticket_id", "verified", "verification_required", "date", "found")}}
             for c in result.get("tool_calls", [])]
    lines = [f"**Decision:** {result.get('decision')}",
             f"**Identity check:** {result.get('order_state', AGENT.state.describe())}",
             f"**Sources cited:** {', '.join(result.get('sources', [])) or '-'}",
             f"**Retrieved:** {', '.join(result.get('retrieved_ids', [])) or '-'}"]
    if result.get("transcript") is not None:
        lines.insert(0, f"**Gnani STT heard:** {result['transcript'] or '(nothing)'}")
    if tools:
        lines.append("**Tool calls:**\n```json\n" + json.dumps(tools, ensure_ascii=False, indent=1) + "\n```")
    return "\n\n".join(lines)


def on_audio(audio_path, history):
    if not audio_path:
        return history, None, "Record something first.", None
    if not SESSION:
        return history, None, "Set GNANI_API_KEY to enable voice.", None
    wav = OUT / f"reply_{len(history)}.wav"
    result = SESSION.turn(audio_path, wav)
    history = history + [{"role": "user", "content": f"🎙️ {result['transcript'] or '(unclear audio)'}"},
                         {"role": "assistant", "content": result["reply"]}]
    return history, str(wav), _panel(result), None


def on_text(message, history):
    if not message.strip():
        return history, None, "", ""
    result = AGENT.respond(message)
    wav = None
    if VOICE:
        wav = OUT / f"reply_{len(history)}.wav"
        VOICE.synthesize(result["reply"], wav, result.get("language", ""))
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": result["reply"]}]
    return history, str(wav) if wav else None, _panel(result), ""


def on_reset():
    AGENT.reset()
    if SESSION:
        SESSION.misses = 0
    return [], None, "New call started.", None


with gr.Blocks(title="ZentroMart VoiceOps") as demo:
    gr.Markdown("## ZentroMart support (VoiceOps 2026)\nSpeak or type. "
                "Speech is handled by Gnani STT/TTS; answers come only from ZentroMart's documents.")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=420)
            mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak to the agent")
            text = gr.Textbox(placeholder="...or type a message and press Enter", show_label=False)
            reply_audio = gr.Audio(label="Agent reply", autoplay=True, interactive=False)
            reset = gr.Button("New call")
        with gr.Column(scale=2):
            panel = gr.Markdown("Agent internals will show here.")
    reset.click(on_reset, outputs=[chat, reply_audio, panel, mic])
    mic.stop_recording(on_audio, [mic, chat], [chat, reply_audio, panel, mic])
    mic.upload(on_audio, [mic, chat], [chat, reply_audio, panel, mic])
    text.submit(on_text, [text, chat], [chat, reply_audio, panel, text])

if __name__ == "__main__":
    demo.launch()
