"""Small text helpers: spoken numbers to digits, ID extraction, speech formatting."""
from __future__ import annotations

import re
from datetime import date

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Spoken digits as STT may return them (English, Romanised Hindi, Devanagari).
_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "shunya": "0", "shoonya": "0", "शून्य": "0",
    "one": "1", "won": "1", "ek": "1", "एक": "1",
    "two": "2", "to": "2", "too": "2", "do": "2", "दो": "2",
    "three": "3", "tree": "3", "teen": "3", "तीन": "3",
    "four": "4", "for": "4", "char": "4", "chaar": "4", "चार": "4",
    "five": "5", "paanch": "5", "panch": "5", "पांच": "5", "पाँच": "5",
    "six": "6", "chhe": "6", "chhah": "6", "che": "6", "छह": "6", "छः": "6",
    "seven": "7", "saat": "7", "सात": "7",
    "eight": "8", "aath": "8", "आठ": "8",
    "nine": "9", "nau": "9", "नौ": "9",
}
_AMBIGUOUS = {"o", "oh", "to", "too", "do", "for", "won", "tree", "che", "char", "ek"}
_REPEAT = {"double": 2, "triple": 3}


def ascii_digits(text: str) -> str:
    return text.translate(_DEVANAGARI_DIGITS)


def spoken_digits_to_numbers(text: str) -> str:
    """Collapse runs of spoken digits ("four five three three") into "4533".

    Ambiguous words such as "to"/"for"/"do" only count when they sit inside a
    run of at least three digit tokens, so ordinary sentences are left alone.
    """
    text = ascii_digits(text)
    tokens = re.findall(r"\w+|[^\w\s]|\s+", text, flags=re.UNICODE)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        run, j, pending_repeat = [], i, 1
        while j < len(tokens):
            tok = tokens[j]
            low = tok.lower()
            if tok.isspace() or tok == "-":
                j += 1
                continue
            if low in _REPEAT:
                pending_repeat = _REPEAT[low]
                j += 1
                continue
            if low in _DIGIT_WORDS:
                run.append((_DIGIT_WORDS[low] * pending_repeat, low))
            elif tok.isdigit():
                run.append((tok * pending_repeat if len(tok) == 1 else tok, tok))
            else:
                break
            pending_repeat = 1
            j += 1
        word_tokens = [w for _, w in run if not w.isdigit()]
        strong = [w for w in word_tokens if w not in _AMBIGUOUS]
        if run and word_tokens and len(run) >= 3 and strong:
            out.append("".join(d for d, _ in run))
            # keep the whitespace that ended the run
            k = j - 1
            while k > i and tokens[k].isspace():
                k -= 1
            out.extend(tokens[k + 1:j])
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return "".join(out)


def normalize_numbers(text: str) -> str:
    return spoken_digits_to_numbers(ascii_digits(text))


_SKU_RE = re.compile(r"\b(ELE|FSH|HOM|BTY|BKS|GRO|INR|DGT)\s*-?\s*(\d{4})\b", re.I)


def find_order_id(text: str) -> str | None:
    """8-digit order IDs, also written as two groups of four ("8978 2837")."""
    text = normalize_numbers(text)
    match = re.search(r"(?<!\d)(\d{4})[\s-]?(\d{4})(?!\d)", text)
    return match.group(1) + match.group(2) if match else None


def find_verification_value(text: str, order_id: str | None = None) -> str | None:
    """A 4-digit phone suffix or 6-digit pincode in the customer's message."""
    text = normalize_numbers(text)
    text = _SKU_RE.sub(" ", text)
    if order_id:
        text = re.sub(rf"{order_id[:4]}[\s-]?{order_id[4:]}", " ", text)
    joined = re.sub(r"(?<=\d)[\s-](?=\d)", "", text)
    for match in re.finditer(r"(?<!\d)(\d{6}|\d{4})(?!\d)", joined):
        value = match.group(1)
        # skip years and dates
        if len(value) == 4 and re.match(r"20[2-3]\d", value) and re.search(
                rf"(january|february|march|april|may|june|july|august|september|october|"
                rf"november|december|\d)\s*,?\s*{value}", joined, re.I):
            continue
        return value
    return None


def find_skus(text: str) -> list[str]:
    text = normalize_numbers(text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return [f"{p.upper()}-{n}" for p, n in _SKU_RE.findall(text)]


# ---------------------------------------------------------------------------
# Speech formatting (STY-3 and Gnani's text-normalisation guide)
# ---------------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
_HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर",
              "अक्टूबर", "नवंबर", "दिसंबर"]


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def human_date(iso: str, lang: str = "en") -> str:
    d = date.fromisoformat(iso)
    if lang == "hi":
        return f"{d.day} {_HI_MONTHS[d.month - 1]} {d.year}"
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def for_speech(text: str, devanagari: bool = False) -> str:
    """Rewrite a written reply so TTS reads it clearly."""
    s = text
    # Markdown and bullets do not belong in speech.
    s = re.sub(r"[*_`#>]+", "", s)
    s = re.sub(r"^\s*[-•]\s*", "", s, flags=re.M)
    s = re.sub(r"\[(?:[A-Z]{2,5}v?\d?-?[\d.]+|products\.csv:[A-Z]+-\d+)\]", "", s)

    # ISO dates -> "the 27th of October 2026"
    def iso(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if devanagari:
            return f"{d} {_HI_MONTHS[mo - 1]} {y}"
        return f"the {_ordinal(d)} of {MONTHS[mo - 1]} {y}"
    s = re.sub(r"\b(20\d\d)-(\d\d)-(\d\d)\b", iso, s)
    if not devanagari:
        month_alt = "|".join(MONTHS)
        s = re.sub(rf"\b(\d{{1,2}})\s+({month_alt})\b",
                   lambda m: f"the {_ordinal(int(m.group(1)))} of {m.group(2)}", s)
    # 8-digit order IDs / long reference numbers in groups of four
    s = re.sub(r"(?<![\d₹,])(\d{8,12})(?!\d)",
               lambda m: " ".join(m.group(1)[i:i + 4] for i in range(0, len(m.group(1)), 4)), s)
    # Currency
    word = "रुपये" if devanagari else "rupees"
    s = re.sub(r"₹\s?([\d,]+)", lambda m: f"{m.group(1).replace(',', '')} {word}", s)
    s = re.sub(r"\bRs\.?\s?([\d,]+)", lambda m: f"{m.group(1).replace(',', '')} {word}", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([.,!?।])", r"\1", s).strip()
    return s


def has_devanagari(text: str) -> bool:
    return bool(re.search(r"[ऀ-ॿ]", text))
