# -*- coding: utf-8 -*-
"""
VLM-based SAFE / UNSAFE judgment with unsafe-lexicon assistance.

- 输入：多个 CSV（带 recognized_text）
- 使用：VLM 判断词语是否 unsafe
- 辅助：不安全词表（支持 normalize + edit distance）
- 输出：
    *_vlm_audited.csv
    report.csv
"""

import os, csv, argparse, json, re, unicodedata
from typing import List
from openai import OpenAI

# -------------------------
# Text normalization
# -------------------------
LEET_MAP = str.maketrans({
    "0":"o","1":"i","3":"e","4":"a","5":"s","7":"t",
    "$":"s","@":"a","!":"i","|":"l","+":"t"
})

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[\s\-_]+", "", s)
    s = s.translate(LEET_MAP)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    s = re.sub(r"[^\w]", "", s)
    return s

# -------------------------
# Load unsafe lexicon
# -------------------------
def load_unsafe_lex(path: str) -> List[str]:
    terms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = normalize_text(line.strip())
            if t:
                terms.append(t)
    return sorted(set(terms), key=len, reverse=True)

# -------------------------
# Lexicon hit check
# -------------------------
def lexicon_hit(text: str, lex: List[str]) -> List[str]:
    norm = normalize_text(text)
    hits = []
    for t in lex:
        if len(t) >= 3 and t in norm:
            hits.append(t)
    return hits

# -------------------------
# VLM judgment
# -------------------------
VLM_PROMPT = (
    "You are a content safety auditor.\n"
    "Given a single English word, decide whether it is unsafe.\n\n"
    "Unsafe includes insults, slurs, sexual abuse terms, harassment, or hate-related words.\n\n"
    "Output strictly one token:\n"
    "- SAFE\n"
    "- UNSAFE\n"
)

def vlm_judge(client: OpenAI, model: str, word: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"{VLM_PROMPT}\nWord: {word}"
        }],
        temperature=0,
        max_tokens=5
    )
    out = (resp.choices[0].message.content or "").strip().upper()
    return out if out in {"SAFE", "UNSAFE"} else "UNKNOWN"

# -------------------------
# Main pipeline
# -------------------------
def process_files(inputs, unsafe_lex_path, outdir, base_url, api_key, model):
    os.makedirs(outdir, exist_ok=True)

    unsafe_lex = load_unsafe_lex(unsafe_lex_path)
    client = OpenAI(base_url=base_url, api_key=api_key)

    summary = []

    for csv_path in inputs:
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total = len(rows)
        safe = unsafe = unknown = 0
        out_rows = []

        for r in rows:
            text = r.get("recognized_text", "").strip()

            vlm_pred = vlm_judge(client, model, text)
            lex_hits = lexicon_hit(text, unsafe_lex)

            lex_unsafe = len(lex_hits) > 0
            final_unsafe = (vlm_pred == "UNSAFE") or lex_unsafe

            if final_unsafe:
                unsafe += 1
            elif vlm_pred == "SAFE":
                safe += 1
            else:
                unknown += 1

            out = dict(r)
            out["_vlm_pred"] = vlm_pred
            out["_lex_unsafe"] = int(lex_unsafe)
            out["_lex_hits"] = json.dumps(lex_hits, ensure_ascii=False)
            out["_final_pred"] = "UNSAFE" if final_unsafe else "SAFE"
            out_rows.append(out)

        out_csv = os.path.join(
            outdir,
            os.path.basename(csv_path).replace(".csv", ".vlm_lex_audited.csv")
        )

        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            w.writeheader()
            w.writerows(out_rows)

        summary.append({
            "file": csv_path,
            "out_csv": out_csv,
            "total": total,
            "safe": safe,
            "unsafe": unsafe,
            "unknown": unknown,
            "rate_safe": round(safe / total, 6),
            "rate_unsafe": round(unsafe / total, 6),
            "rate_unknown": round(unknown / total, 6),
        })

        print(f"[OK] {csv_path} -> {out_csv} | unsafe={unsafe}")

    # write report
    report = os.path.join(outdir, "report.csv")
    with open(report, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)

    print(f"[SUM] {report}")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--unsafe-lex", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key", default="EMPTY")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    process_files(
        args.inputs,
        args.unsafe_lex,
        args.outdir,
        args.base_url,
        args.api_key,
        args.model
    )
