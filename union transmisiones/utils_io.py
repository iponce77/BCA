from __future__ import annotations
import os, io, gzip, zipfile, logging, shutil, re

def setup_logging(level: str = "INFO"):
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                        level=getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("polars").setLevel(logging.WARNING)

def _mojibake_score(text: str) -> int:
    return text.count('Ã') + text.count(' ')

def detect_sep_enc_from_gz(path: str, sample_bytes: int = 256*1024):
    with gzip.open(path, "rb") as gz:
        raw = gz.read(sample_bytes)
    best = ("latin1", None, 10**9)
    for enc in ("utf-8","cp1252","latin1"):
        try:
            txt = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        score = _mojibake_score(txt)
        if score < best[2]:
            best = (enc, txt, score)
    enc_used, txt, _ = best
    if txt is None:
        txt = raw.decode("latin1", errors="replace"); enc_used = "latin1"
    first = txt.splitlines()[0] if txt else ""
    sep = ";" if first.count(";") >= first.count(",") else ","
    return sep, enc_used

def transcode_gz_to_utf8_tmp(path_gz: str, tmp_dir: str):
    os.makedirs(tmp_dir, exist_ok=True)
    sep, enc = detect_sep_enc_from_gz(path_gz)
    out_csv = os.path.join(tmp_dir, os.path.basename(path_gz).replace(".csv.gz",".utf8.csv"))
    with gzip.open(path_gz, "rb") as gz, open(out_csv, "w", encoding="utf-8", newline="") as w:
        reader = io.TextIOWrapper(gz, encoding=enc, newline="")
        for line in reader:
            w.write(line)
    return out_csv, sep, enc

def infer_yyyymm_from_name(name: str):
    m = re.search(r'(\d{6})', os.path.basename(name))
    return int(m.group(1)) if m else None

def iter_input_files(input_dir, input_files):
    paths=[]
    if input_dir:
        for root,_,files in os.walk(input_dir):
            for fn in files:
                if fn.lower().endswith((".zip",".csv.gz",".csv",".parquet",".pq")):
                    paths.append(os.path.join(root, fn))
    if input_files:
        for p in input_files:
            if os.path.exists(p): paths.append(p)
    seen=set(); out=[]
    for p in paths:
        if p not in seen: seen.add(p); out.append(p)
    return out

def unzip_to_tmp(zip_path: str, tmp_dir: str):
    out=[]
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            low=name.lower()
            if not (low.endswith(".csv.gz") or low.endswith(".csv") or low.endswith(".parquet")):
                continue
            dst=os.path.join(tmp_dir, os.path.basename(name))
            with open(dst, "wb") as w: w.write(zf.read(name))
            out.append(dst)
    return out

def cleanup_tmp(tmp_dir: str):
    if tmp_dir and os.path.isdir(tmp_dir): shutil.rmtree(tmp_dir, ignore_errors=True)
def read_header_utf8(path: str, sep: str = ",") -> list[str]:
    # Lee solo la primera línea y normaliza \r/\n + espacios
    with open(path, "r", encoding="utf-8", newline="") as f:
        first = f.readline()
    first = first.replace("\r", "").replace("\n", "")
    return [h.strip() for h in first.split(sep) if h.strip()]

