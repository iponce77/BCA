#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
from segments import load_segment_map, infer_segment

def read_any(path: Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"): return pd.read_parquet(path)
    if p.endswith(".csv"):     return pd.read_csv(path, low_memory=False)
    if p.endswith(".xlsx") or p.endswith(".xls"): return pd.read_excel(path)
    raise SystemExit(f"Formato no soportado: {path}")

def write_any(df: pd.DataFrame, path: Path):
    p = str(path).lower()
    if p.endswith(".parquet"): return df.to_parquet(path, index=False)
    if p.endswith(".csv"):     return df.to_csv(path, index=False)
    if p.endswith(".xlsx") or p.endswith(".xls"): return df.to_excel(path, index=False)
    raise SystemExit(f"Formato no soportado: {path}")

def main():
    if len(sys.argv) < 3:
        print("Uso: add_segmento.py <input.{parquet|csv|xlsx}> <output.{parquet|csv|xlsx}> [segment_map.csv]")
        sys.exit(1)

    in_path  = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    map_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("segment_map.csv")

    segmap = load_segment_map(str(map_path))
    df = read_any(in_path)

    # Detectar columnas reales (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    mk_col = cols.get("make_clean")
    mb_col = cols.get("modelo_base")

    if not mk_col or not mb_col:
        raise SystemExit("El dataset de entrada debe contener 'make_clean' y 'modelo_base'.")

    df["segmento"] = [infer_segment(mk, mb, segmap) for mk, mb in zip(df[mk_col], df[mb_col])]
    write_any(df, out_path)
    print(f"OK → {out_path} | segmento nulos: {int(df['segmento'].isna().mean()*100)}%")

if __name__ == "__main__":
    main()
