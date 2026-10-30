# ZentroMart VoiceOps agent (VoiceOps 2026 capstone)

A customer-support agent for ZentroMart that answers from the company's documents, holds a verified multi-turn conversation, and takes phone-style voice calls through **Gnani STT and TTS**.

- **GitHub:** https://github.com/parthpandey-16/voiceops
- **Report:** [`report.pdf`](report.pdf) · **Demo:** [`demo.mp4`](demo.mp4) · **Sample outputs:** [`sample_outputs/`](sample_outputs/)

```
customer audio ─▶ Gnani STT ─▶ agent (LLM + hybrid RAG + tools + memory) ─▶ Gnani TTS ─▶ spoken reply
```

## 1. Install

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # then add your keys to .env
```

Put the extracted dataset folder `Capstone_Project_Dataset_FTTT_2026/` next to this README, or set `DATASET_DIR` in `.env` to wherever it is. The dataset is not included in this repo, as the rules ask.

Keys go in `.env`, which git ignores:

| Variable | What it's for |
| --- | --- |
| `GEMINI_API_KEY` (or `OPENAI_API_KEY` / `GROQ_API_KEY` plus `LLM_PROVIDER`) | The LLM. Any OpenAI-compatible API works. |
| `GNANI_API_KEY` | Gnani speech-to-text and text-to-speech (from app.gnani.ai/voice). |

## 2. Run command (used by the organisers)

```bash
python run.py --input path/to/audio_folder --output outputs/run1
```

For every audio file it saves the reply as **text** (`.reply.txt`), as **speech** (`.reply.wav`, Gnani TTS), and a `.json` file holding the transcript, the decision, the tool calls, the **cited document sections** and **every retrieved chunk** with its text and score. It also writes `results.csv` and `results.jsonl` covering every file.

How files become calls (each call gets its own memory):

- each **sub-folder** is one conversation, with its files taken in natural order;
- flat files named `<id>_turn<N>.wav` (or `<id>_t<N>`) form one conversation per `<id>`;
- a `manifest.csv` with the columns `file, conversation_id, turn` overrides both;
- any other file is a single-turn call.

`--mode single` or `--mode conversation` forces one behaviour. Supported audio formats are `.wav .mp3 .flac .ogg .m4a .aac`. Clips longer than 55 s are split, because the Gnani REST STT endpoint accepts up to 60 s.

## 3. Other commands

```bash
python -m pytest tests -q                       # offline tests, no keys needed
python evaluate.py --step all                   # Step 1 vs Step 2 (+ASR text) and Step 3 scores -> results/
python scripts/build_report.py                  # report.pdf from results/metrics.json
python app.py                                   # live browser demo: microphone in, speech out
python scripts/make_practice_audio.py           # optional: Gnani TTS test audio from the practice set
```

`python evaluate.py --step ablation` adds the no-query-rewrite run used in the report. Evaluation results are resumable: if a free-tier rate limit stops a run, run the command again and it continues from where it stopped.

## 4. How it maps to the four steps and the rubric

| Step / rubric row | Where | What it does |
| --- | --- | --- |
| **Step 1** prompt-only baseline | `src/voiceops/agent.py: BaselineAgent`, `prompts.py: BASELINE_SYSTEM` | System prompt sets the role, limits and AI disclosure. No documents. Scored on the 52 practice questions. |
| **Step 2** RAG · *Correct grounded answers (25)* | `kb.py`, `retriever.py`, `agent.py: SupportAgent` | One chunk per `[SECTION-ID]`, keeping `doc_type`, `status` and `effective_from`. The agent answers only from the sections and catalog rows it is given, follows KB-00.1 precedence, and says "I don't have that information" plus offers a colleague when nothing covers the question. |
| *Retrieval quality (15)* | `retriever.py` | Hybrid **BM25 + multilingual dense embeddings** fused with RRF; an LLM **query rewrite** turns Hindi, Hinglish and follow-ups into a standalone English query; LLM-generated **document expansion** (customer-style paraphrases per section, cached); cross-reference following (`REF-1`, `DWM-1`…); a light KB-00.1 authority prior. Every reply reports the section IDs it used and every chunk retrieved. |
| **Step 3** · *Conversation handling (20)* | `agent.py: ConversationState`, `orders.py` | Remembers order ID, item and topic across turns; resolves follow-ups; asks one question when a detail is missing. Identity check is **deterministic code**: `lookup_order(order_id)` when an ID appears, then `lookup_order(order_id, verification_value)` with one retry, then lock (IDV-1 to IDV-3). Verification digits are never shown to the LLM, logged, or put in ticket summaries. `create_ticket` is called before any handover, and a guard makes sure the agent never claims a ticket that the tool did not create (ESC-2, ESC-4). Exact date maths runs in a `date_calc` tool (business days skip Sundays and `holidays.csv`). |
| *Your own evaluation (10)* | `src/voiceops/evaluate.py`, `results/` | The same scorer for Step 1, Step 2, Step 2 on ASR-style text, Step 3 turns, and a no-rewrite ablation: decision, `must_include` / `must_not_include`, tool calls, retrieval recall and precision, and breakdowns by intent and language. |
| **Step 4** · *Voice works end to end (10)* | `voice.py`, `run.py` | Gnani STT (`format="transcribe"`, so spoken digits become numbers) → the same agent → Gnani TTS (`timbre-v2.5`, Kaveri for English, Nalini for Hindi, Poorvi for Hinglish). Every file gets a reply even if a call fails. |
| *Voice experience (10)* | `voice.py`, `textnorm.py`, `prompts.py: VOICE_RULES` | Two-sentence voice replies; dates are read as "the 27th of October 2026", order IDs in groups of four, "₹49" as "49 rupees", with no markdown or brackets. Unclear or empty audio gets "Sorry, I didn't catch that", and a second miss offers a human colleague. Spoken digits ("four five three three") are understood too. |
| *Report and demo (10)* | `report.pdf`, `demo.mp4`, `app.py` | |

## 5. Rules compliance

- Speech uses **Gnani only**. There is no other STT or TTS library anywhere in the code, and no backup speech service. Fallbacks only switch to another Gnani voice or language.
- Only the synthetic ZentroMart data is used. No answers are hard-coded: every reply is generated by the agent from retrieved documents, catalog rows and tool results.
- No keys in code. `.env` is git-ignored, and `.env.example` has no values.

## 6. Repository layout

```
run.py                     batch audio run command
app.py                     browser demo (Gradio)
src/voiceops/
  config.py                settings from env/.env
  kb.py                    KB parsing, one chunk per section ID
  retriever.py             BM25 + dense + RRF + expansion
  catalog.py               products.csv / pincodes.csv lookups
  dates.py                 business-day maths (KB-00.3)
  orders.py                dataset tools bridge, secret scrubbing
  prompts.py               every prompt
  agent.py                 BaselineAgent (Step 1), SupportAgent (Steps 2-4)
  voice.py                 Gnani STT/TTS, unclear-audio recovery
  textnorm.py              ID extraction, spoken digits, speech formatting
  evaluate.py              scoring and evaluation runs
scripts/                   report builder, practice-audio maker, packager
tests/                     offline tests
results/                   evaluation outputs and metrics.json
sample_outputs/            run.py outputs on sample audio
```
