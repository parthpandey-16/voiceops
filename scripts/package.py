#!/usr/bin/env python
"""Build the submission ZIP in the layout the brief asks for.

    python scripts/package.py --team "your_team_name"   ->  dist/your_team_name_voiceops.zip

Leaves out .env, virtual environments, caches of downloaded models and the dataset.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["README.md", ".env.example", "requirements.txt", "run.py", "app.py", "evaluate.py",
           "report.pdf", "demo.mp4", ".gitignore", "src", "scripts", "tests", "results", "sample_outputs"]
CACHE_FILES = ["expansions_*.json", "vectors_*.npy"]  # small, derived from the KB; saves LLM calls
SKIP = re.compile(r"(__pycache__|\.pyc$|\.env$|\.venv|/venv/|Capstone_Project_Dataset|\.DS_Store|"
                  r"tickets_created\.csv)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", required=True)
    args = parser.parse_args()
    team = re.sub(r"\W+", "_", args.team.strip()).strip("_").lower()
    out = ROOT / "dist" / f"{team}_voiceops.zip"
    out.parent.mkdir(exist_ok=True)
    missing = [n for n in ("report.pdf", "demo.mp4", "sample_outputs") if not (ROOT / n).exists()]
    files = []
    for name in INCLUDE:
        path = ROOT / name
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files += [p for p in path.rglob("*") if p.is_file()]
    for pattern in CACHE_FILES:
        files += list((ROOT / ".cache").glob(pattern))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            if not SKIP.search(rel):
                zf.write(path, f"{team}_voiceops/{rel}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if missing:
        print("WARNING: not found yet:", ", ".join(missing))


if __name__ == "__main__":
    main()
