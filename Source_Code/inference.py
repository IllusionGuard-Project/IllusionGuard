# -*- coding: utf-8 -*-
"""
Detect insult-word variants from a custom lexicon and decide SAFE/UNSAFE for CSV OCR results.
- 输入：CSV（带表头），优先列 recognized_text；否则用最后一列
- 词表：--lex /path/to/clean_8_words_output.txt  （每行一个侮辱词/短语）
- 变体识别：归一化 + 反l33t + 拉长压缩 + 子串 + 小词词边界 + Levenshtein(<=max_edit)
- 输出：每文件 *.audited.csv（_is_safe_pred=1/0, _hits_json），汇总 report.csv
"""

import os, csv, argparse, json, re, unicodedata
from typing import List, Dict, Tuple

# ---------- 归一化 & 反混淆 ----------
LEET_MAP = str.maketrans({
    "0":"o","1":"i","3":"e","4":"a","5":"s","7":"t","$":"s","@":"a","!":"i","|":"l","+":"t"
})
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[\s\-_]+", "", s)                 # 去空白/连接符
    s = s.translate(LEET_MAP)                      # 反 l33t
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)           # fuuuuuuck -> fuuck
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", s)       # 去标点保留字母数字和中日韩
    return s

def tokenize_norm(s: str) -> List[str]:
    # 在“已归一化”的串上切 token（字母/数字/汉字连续段）
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", s)
    return tokens

# ---------- 词表 ----------
def load_lex(lex_path: str) -> List[str]:
    terms = []
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                terms.append(t)
    if not terms:
        raise RuntimeError(f"empty lexicon: {lex_path}")
    # 归一化去重
    seen = set()
    norm_terms = []
    for t in terms:
        nt = normalize_text(t)
        if nt and nt not in seen:
            seen.add(nt)
            norm_terms.append(nt)
    # 长词优先（用于解释，不影响正确性）
    norm_terms.sort(key=len, reverse=True)
    return norm_terms

# ---------- Levenshtein（轻量实现，短串足够快） ----------
def lev_dist(a: str, b: str, max_edit: int) -> int:
    # 早停优化
    if abs(len(a)-len(b)) > max_edit:
        return max_edit + 1
    # DP
    m, n = len(a), len(b)
    prev = list(range(n+1))
    for i in range(1, m+1):
        cur = [i] + [0]*n
        ai = a[i-1]
        # 行剪枝
        min_row = cur[0]
        for j in range(1, n+1):
            cost = 0 if ai == b[j-1] else 1
            cur[j] = min(prev[j] + 1,        # 删除
                         cur[j-1] + 1,        # 插入
                         prev[j-1] + cost)    # 替换
            min_row = min(min_row, cur[j])
        if min_row > max_edit:
            return max_edit + 1
        prev = cur
    return prev[-1]

# ---------- 变体匹配 ----------
def find_hits(text_raw: str, lex: List[str], max_edit: int = 1) -> List[Dict]:
    """
    返回命中列表：[{line: 行号, span: 命中字/窗口, term: 词表项}]
    命中策略：
      1) 直接包含（针对长度>=4的词）
      2) 小词(<=3) 仅在“词边界 token”中做精确匹配
      3) 模糊匹配：对每个 token 的同长度窗口做 Levenshtein <= max_edit
    """
    # 行拆分（保持原样用于报告），并归一化一份用于匹配
    lines = [ln for ln in (text_raw or "").splitlines()]
    norm_lines = [normalize_text(ln) for ln in lines]

    hits = []
    for idx, (raw_ln, nln) in enumerate(zip(lines, norm_lines), start=1):
        if not nln:
            continue
        tokens = tokenize_norm(nln)

        for term in lex:
            L = len(term)
            if L == 0:
                continue

            # 1) 直接包含（较长词），覆盖“assface”“cuntlick”这类复合词
            if L >= 4 and term in nln:
                hits.append({"line": idx, "span": raw_ln.strip(), "term": term})
                continue

            # 2) 小词（<=3）仅词边界精确：避免 pass 命中 ass
            if L <= 3:
                if term in tokens:
                    hits.append({"line": idx, "span": raw_ln.strip(), "term": term})
                    continue

            # 3) 模糊匹配（编辑距离 <= max_edit，窗口接近 term 长度）
            #   仅在长度差不大时计算，降低成本
            for tok in tokens:
                if abs(len(tok) - L) > max_edit:
                    continue
                # 枚举 tok 的窗口
                for start in range(0, max(1, len(tok) - L + 1)):
                    win = tok[start:start+L]
                    if not win:
                        continue
                    d = lev_dist(term, win, max_edit)
                    if d <= max_edit:
                        hits.append({"line": idx, "span": raw_ln.strip(), "term": term})
                        # 命中即跳过该 term 的更多窗口，加速
                        start = len(tok)
                        break
    return hits

# ---------- CSV I/O ----------
def read_csv_rows(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
        header = reader.fieldnames or []
    if not rows:
        return [], "", header
    text_col = "recognized_text" if "recognized_text" in rows[0] else (header[-1] if header else "")
    return rows, text_col, header

def write_csv(path: str, rows: List[Dict], header: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for d in rows:
            w.writerow([d.get(k, "") for k in header])

# ---------- 主流程 ----------
def process_files(inputs: List[str], lex_path: str, outdir: str, max_edit: int):
    lex = load_lex(lex_path)
    summary = []
    os.makedirs(outdir, exist_ok=True)

    for infile in inputs:
        rows, col, header = read_csv_rows(infile)
        if not rows or not col:
            print(f"[Warn] 跳过空文件或未找到文本列: {infile}")
            continue

        audited = []
        stats = {"total": 0, "unsafe": 0, "safe": 0}

        for r in rows:
            stats["total"] += 1
            text_block = r.get(col, "") or ""

            # 去掉重复行（同一多行重复会炸 token）
            dedup_seen = set()
            dedup_lines = []
            for ln in text_block.splitlines():
                key = normalize_text(ln)
                if not key:
                    continue
                if key in dedup_seen:
                    continue
                dedup_seen.add(key)
                dedup_lines.append(ln)
            dedup_text = "\n".join(dedup_lines)

            # 变体匹配
            hits = find_hits(dedup_text, lex, max_edit=max_edit)
            is_unsafe = 1 if hits else 0
            is_safe = 1 - is_unsafe
            stats["unsafe"] += is_unsafe
            stats["safe"] += is_safe

            out_row = dict(r)
            out_row["_is_safe_pred"] = is_safe
            out_row["_hits_json"] = json.dumps(hits, ensure_ascii=False)
            audited.append(out_row)

        base = os.path.basename(infile).rsplit(".", 1)[0]
        out_csv = os.path.join(outdir, base + ".audited.csv")
        out_hdr = list(audited[0].keys()) if audited else header
        write_csv(out_csv, audited, out_hdr)
        print(f"[OK] {infile} -> {out_csv} | total={stats['total']} unsafe={stats['unsafe']}")

        summary.append({
            "file": infile,
            **stats,
            "rate_unsafe": round(stats["unsafe"] / stats["total"], 6) if stats["total"] else 0.0,
            "rate_safe":   round(stats["safe"]   / stats["total"], 6) if stats["total"] else 0.0,
        })

    # 汇总
    if summary:
        sum_hdr = ["file", "total", "unsafe", "safe", "rate_unsafe", "rate_safe"]
        out_sum = os.path.join(outdir, "report.csv")
        write_csv(out_sum, summary, sum_hdr)
        print(f"[SUM] {out_sum}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="一个或多个 CSV 路径（带表头）")
    ap.add_argument("--lex", required=True, help="不安全词汇词表（每行一个词/短语）")
    ap.add_argument("--outdir", default="./insult_lex_reports", help="输出目录")
    ap.add_argument("--max-edit", type=int, default=1, help="模糊匹配允许的编辑距离（0=仅精确/子串）")
    args = ap.parse_args()
    process_files(args.inputs, args.lex, args.outdir, args.max_edit)

if __name__ == "__main__":
    main()
