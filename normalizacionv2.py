#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalizacionv2.py
====================================================
Extensión del normalizador fusión (v2+v3) con ajustes de ENTRADA SOLO para MERCEDES:

1) **"Submodelo primero"**: si existe una columna 'submodelo' (o variantes) y, para
   MERCEDES BENZ, aporta más información que 'modelo', se usará preferentemente.

2) **Preclasificación segura** (A/B/C/E/S + 2-3 dígitos al INICIO → CLASE X).
   Además, si aparece 'COUPE' en el texto y existe 'CLASE X COUPE' en whitelist,
   se usa esa familia. Esta preclasificación es puramente de input (no altera el
   motor), y es **sólo** para MERCEDES BENZ.

El resto de marcas y bases de datos quedan **intactas**.

Uso (igual que el original):
    python normalizacion_fusion_plug_prod.py --input <in> --whitelist whitelist.xlsx --out <out>
"""

from __future__ import annotations
import json, re, sys, argparse
from typing import Dict, Set, Tuple, Optional
from pathlib import Path

# Dependencias locales
sys.path.append("/mnt/data")
import pandas as pd
import normalizacionv2_legacy as v2
import normalizacionv3 as v3

# ---------------------------
# Utils
# ---------------------------
def _norm(s: Optional[str]) -> str:
    if s is None: return ""
    return v3.normalize_text(s)

def _load_weights(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    defaults = {
        "default": {"recall":0.45, "pref":0.20, "length":0.05, "subset":0.10, "digits":0.05,
                    "guard":0.10, "v2flag":0.15, "family":0.10, "lex":0.10,
                    "first_token":0.20, "pref_pos":0.15, "present":0.25, "coupe":0.0}
    }
    if not path:
        return defaults
    p = Path(path)
    if not p.exists():
        return defaults
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = {"default": defaults["default"], **data}
    return data

def _load_input(input_path: str, sheet=None) -> pd.DataFrame:
    p = Path(input_path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(p, sheet_name=(sheet or 0), engine="openpyxl")
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="latin-1")

def _save_output(df: pd.DataFrame, out_path: str):
    p = Path(out_path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        df.to_parquet(p, index=False)
    elif suf in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
    else:
        df.to_csv(p, index=False)

def _detect_cols(df: pd.DataFrame, make_arg: Optional[str], model_arg: Optional[str]) -> Tuple[str, str, Optional[str]]:
    lower = {c.lower(): c for c in df.columns}
    make_col  = make_arg  or lower.get("make") or lower.get("marca") or lower.get("make_clean")
    model_col = model_arg or lower.get("model") or lower.get("modelo")
    # Variantes de submodelo que a veces aparecen
    sub_candidates = ["submodelo","sub-modelo","submodel","sub_model","sub-mod","sub_mod","sub"]
    sub_col = None
    for k in sub_candidates:
        if k in lower:
            sub_col = lower[k]
            break
    if not make_col:
        # plan B
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("make","marca","make_clean"):
                make_col = c; break
    if not model_col:
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("model","modelo"):
                model_col = c; break
    if not make_col or not model_col:
        raise SystemExit("No encuentro columnas de marca/modelo. Usa --make-col/--model-col.")
    return make_col, model_col, sub_col

def _choose_text_mercedes(marca, modelo, submodelo, wl3: Dict[str, Set[str]]) -> str:
    """Submodelo primero (si aporta más) + preclasificación segura, sólo MERCEDES."""
    mk_norm = v3.normalize_brand(marca)
    md = str(modelo) if modelo is not None else ""
    sub = str(submodelo) if submodelo is not None else ""
    if mk_norm != "MERCEDES BENZ":
        return md

    md_u  = v3.normalize_text(md)
    sub_u = v3.normalize_text(sub)
    use = sub if (sub_u and (len(sub_u) > len(md_u) or not md_u)) else (md or sub)

    up = v3.normalize_text(use)

    # Preclasificación: ANCLADO AL INICIO para evitar falsos positivos (p.ej. '... S213 ...')
    m0 = re.match(r'^([ABCES])[-\s]?\d{2,3}', up)
    if m0:
        fam = f"CLASE {m0.group(1)}"
        if "COUPE" in up and (fam + " COUPE") in wl3.get("MERCEDES BENZ", set()):
            return fam + " COUPE"
        return fam
    return use

# ---------------------------
# Motor (delegamos en tu fusión actual)
# ---------------------------
from normalizacion_fusion_plug_prod import normaliza_modelo_fusion as fuse_normalize

# ---------------------------
# CLI
# ---------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(description="Normalizador fusión v2+v3 con 'submodelo primero' sólo MERCEDES")
    ap.add_argument("positional_input", nargs="?", help="(compat) input si no usas --input")
    ap.add_argument("--input", help="Ruta de entrada (.xlsx/.xls/.csv/.parquet)")
    ap.add_argument("--sheet", default=None, help="Hoja si Excel")
    ap.add_argument("--make-col", default=None, help="Sobrescribe columna marca")
    ap.add_argument("--model-col", default=None, help="Sobrescribe columna modelo")
    ap.add_argument("--whitelist", required=True, help="Ruta XLSX whitelist")
    ap.add_argument("--output_excel", help="(compat) salida .xlsx")
    ap.add_argument("--out", help="Salida si no usas --output_excel (admite .xlsx/.csv/.parquet)")
    ap.add_argument("--out-col", default=None, help="Alias legacy. Se ignora; siempre 'modelo_base'")
    ap.add_argument("--outcol", default=None, help="Alias legacy alternativo. Se ignora igualmente.")
    ap.add_argument("--weights", default="weights.json", help="Pesos JSON (opcional)")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    _ = args.out_col  # ignorado intencionalmente (legacy)
    _ = getattr(args, "outcol", None)
    input_path = args.input or args.positional_input
    if not input_path:
        raise SystemExit("Debes indicar un input: positional o --input")

    out_path = args.output_excel or args.out or input_path  # sobreescribe si no hay salida

    df = _load_input(input_path, sheet=getattr(args, "sheet", None))

    make_col, model_col, sub_col = _detect_cols(df, args.make_col, args.model_col)

    # Cargar recursos
    wl3 = v3.carga_whitelist(args.whitelist)
    disp3 = v3._v3_build_display_map(args.whitelist)
    wl2  = v2.carga_whitelist(args.whitelist)
    weights = _load_weights(args.weights)

    # Ejecutar fila a fila
    out_best = []
    for _, r in df.iterrows():
        mk = r.get(make_col)
        md = r.get(model_col)
        sub = r.get(sub_col) if sub_col else None
        md_use = _choose_text_mercedes(mk, md, sub, wl3)
        try:
            best, _, _ = fuse_normalize(mk, md_use, wl3, disp3, wl2, weights)
        except Exception:
            best = None
        out_best.append(best)

    out = df.copy()
    out["modelo_base"] = out_best
    out["make_clean"]  = out[make_col].apply(v3.normalize_brand)
    _save_output(out, out_path)
    print(f"OK -> {out_path} | usando {make_col}/{model_col}" + (f" + {sub_col}" if sub_col else ""))

if __name__ == "__main__":
    main()
