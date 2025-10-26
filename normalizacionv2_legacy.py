
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalizacionv2.py  (compatible con el script antiguo + autodetección de columnas)
==========================================================================
CLI compatible con el anterior:
    python normalizacionv2.py <input.xlsx> --whitelist whitelist.xlsx --output_excel <output.xlsx>

También admite:
    --input, --sheet, --make-col, --model-col, --out, --out-col

Novedad:
- Si NO existen columnas 'make'/'model', buscará automáticamente 'marca'/'modelo'.
- Pensado para tus 3 bases (Autoscout/Milanuncios, Ganvam, DGT).

Salida por defecto:
- Escribe/actualiza la columna **modelo_base** (idéntico al script antiguo).
"""

import argparse
import sys
import re
import unicodedata
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, Optional
from pathlib import Path


# ---------------- Utils ----------------

def strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = strip_accents(s)
    s = re.sub(r'[\/_,.;:]+', ' ', s)
    s = re.sub(r'\b(CX)[\s\-]?(\d)\b', r'\1-\2', s)   # CX-5, CX-3
    s = re.sub(r'\bID[\s\-]?(\d)\b', r'ID.\1', s)     # ID.3, ID.4
    s = s.replace('-', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------------- Whitelist ----------------

def carga_whitelist(path: str) -> Dict[str, Set[str]]:
    df = pd.read_excel(path)
    marca_col = next((c for c in df.columns if str(c).lower().startswith("marca")), df.columns[0])
    modelo_col = next((c for c in df.columns if str(c).lower().startswith("modelo")), df.columns[1])
    wl = defaultdict(set)
    for _, row in df.iterrows():
        mk = normalize_text(row[marca_col])
        md = normalize_text(row[modelo_col])
        if mk and md:
            wl[mk].add(md)
    return wl

# ---------------- Aliases & Noise ----------------

BRAND_ALIAS = {
    "MERCEDES": "MERCEDES BENZ",
    "MERCEDES BENZ": "MERCEDES BENZ",
    "MERCEDES-BENZ": "MERCEDES BENZ",
    "MB": "MERCEDES BENZ",
}

MODEL_ALIAS: Dict[str, Dict[str, str]] = defaultdict(dict)
MODEL_ALIAS["MERCEDES BENZ"].update({
    "EVITO": "E VITO",
    "ESPRINTER": "E SPRINTER",
    "ECITAN": "E CITAN",
})

TOKENS_RUIDO_GLOBAL = {
    "AUTOMOBILES","EDITION","BUSINESS","PACK","LINE","SPORT","SPORTS",
    "AUTOTRONIC","AUTOMATIC","S&S","MY20","CV","CVS","CVV","HP","PS",
    "DIESEL","BENZINA","GASOLINA","PETROL","HYBRID","ELECTRIC","ELEKTROMOTOR",
    "LANG"
}

RUIDO_EXCEPTIONS = {
    "JEEP":{"GRAND"},
    "PORSCHE":{"GT","GTS","GT3","GT4"}
}

# ---------------- Core ----------------

def _load_input(input_path: str, sheet=None) -> pd.DataFrame:
    p = Path(input_path)
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)  # requiere pyarrow
    if suf in (".xlsx", ".xls"):
        # mantiene compat con --sheet si lo tienes en argparse
        return pd.read_excel(p, sheet_name=(sheet or 0), engine="openpyxl")
    # CSV: prueba utf-8 y cae a latin-1 si viene en cp1252 (0x92, etc.)
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="latin-1")
        
def apply_alias_inline(text: str, alias_map: Dict[str, str]) -> str:
    if not alias_map: return text
    out = text
    for src, dst in alias_map.items():
        pattern = rf'(?:(?<=^)|(?<=[\s])){re.escape(src)}(?:(?=$)|(?=[\s]))'
        out = re.sub(pattern, dst, out)
    return re.sub(r'\s+', ' ', out).strip()

def tokens_of(text: str):
    return set(text.split()) if text else set()

def candidates_for_brand(brand: str, wl: Dict[str, Set[str]]):
    return wl.get(brand, set())

def score_candidate(cand: str, model_norm: str, t_model, t_ruido) -> int:
    score=0
    if cand==model_norm: score+=100
    if cand in model_norm: score+=60
    t_cand=tokens_of(cand)
    if t_cand and t_cand.issubset(t_model): score+=40
    score+=min(len(cand),40)//2
    if any(tok.isdigit() for tok in t_cand) or re.search(r'\d', cand):
        if re.search(r'\d', model_norm): score+=15
    if t_cand & t_ruido: score-=10
    return score

def normalize_brand(brand: str) -> str:
    b = normalize_text(brand)
    return {"MERCEDES": "MERCEDES BENZ", "MERCEDES-BENZ": "MERCEDES BENZ"}.get(b, b)

def normaliza_modelo(marca: str, modelo: str, whitelist_global: Dict[str, Set[str]]):
    """Devuelve el modelo normalizado o None (compat con script antiguo)."""
    marca_n = normalize_brand(marca)
    modelo_n = normalize_text(modelo)

    if marca_n not in whitelist_global:
        return None

    modelo_n = apply_alias_inline(modelo_n, MODEL_ALIAS.get(marca_n, {}))

    # --- PATCH ESPECÍFICO MERCEDES ---
    if marca_n == "MERCEDES BENZ":
        # 1) "C KLASSE" / "KLASSE C" -> "CLASE C"; también cubre "CLA KLASSE"
        modelo_n = re.sub(r'\b([A-Z]{1,3})\s+KLASSE\b', r'CLASE \1', modelo_n)
        modelo_n = re.sub(r'\bKLASSE\s+([A-Z]{1,3})\b', r'CLASE \1', modelo_n)
        modelo_n = modelo_n.replace(" KLASSE", " CLASE")

        # 2) Si hay patrón "C 220" / "E300" etc., prioriza la clase
        m = re.search(r'\b([ABCESG])\s?\d{2,3}\b', modelo_n)
        cands_mb = candidates_for_brand(marca_n, whitelist_global)
        if m and f"CLASE {m.group(1)}" in cands_mb:
            return f"CLASE {m.group(1)}"
    # --- FIN PATCH ---
    # --- PATCH ESPECÍFICO BMW ---
    if marca_n == "BMW":
        # 1) Patrones de referencia de serie a partir de códigos 118D, 318D, 520D...
        m = re.search(r'\b([12345678])\d{2}[A-Z]?\b', modelo_n)
        if m:
            serie = m.group(1)
            cand = f"SERIE {serie}"
            if cand in whitelist_global.get(marca_n, set()):
                return cand
        # 2) Mapear eléctricos i3, i4, i5, i7, iX, iX1, iX2, iX3
        for ev in ["I3","I4","I5","I7","IX","IX1","IX2","IX3"]:
            if re.search(rf"\b{ev}\b", modelo_n):
                if ev in whitelist_global.get(marca_n, set()):
                    return ev
        # 3) Si aparece 'TOURING' o 'ACTIVE TOURER' y hay SERIE 2/3/5 en candidatos, quedarse con la serie detectada por el punto 1
        if "TOURING" in modelo_n or "ACTIVE TOURER" in modelo_n:
            if m:
                serie = m.group(1)
                cand = f"SERIE {serie}"
                if cand in whitelist_global.get(marca_n, set()):
                    return cand
    # --- FIN PATCH BMW ---



    toks = modelo_n.split()
    t_ruido = TOKENS_RUIDO_GLOBAL.copy()
    # --- Ruido extra BMW (solo BMW) ---
    if marca_n == "BMW":
        t_ruido |= {"XDRIVE","STEPTRONIC","AUT","AUTOMATIC","DCT","TOURING","GRAN","COUPE","GRAN COUPE","GRAN TURISMO","PACK","LINE","MHEV"}


    # --- Ruido extra Mercedes (no afecta a otras marcas) ---
    if marca_n == "MERCEDES BENZ":
        t_ruido |= {"4MATIC","9G","9GTRONIC","TRONIC","BLUETEC","BLUEEFFICIENCY","KOMBI","KOMPAKT","TOURER","AMG"}

    if marca_n in RUIDO_EXCEPTIONS:
        t_ruido = t_ruido - RUIDO_EXCEPTIONS[marca_n]

    # Reglas por marca
    if marca_n == "PEUGEOT":
        for tok in toks:
            if tok.isdigit() and tok in whitelist_global["PEUGEOT"]:
                return tok
    if marca_n == "FIAT":
        if "500" in toks and "500" in whitelist_global["FIAT"]:
            return "500"
    if marca_n == "MERCEDES BENZ":
        cands = candidates_for_brand(marca_n, whitelist_global)
        if "E" in toks and "VITO" in toks and "E VITO" in cands: return "E VITO"
        if "E" in toks and "SPRINTER" in toks and "E SPRINTER" in cands: return "E SPRINTER"
        if "E" in toks and "CITAN" in toks and "E CITAN" in cands: return "E CITAN"
        if ("E" in toks and "VITO" in toks) or "EVITO" in toks:
            return "VITO" if "VITO" in cands else None
        if ("E" in toks and "SPRINTER" in toks) or "ESPRINTER" in toks:
            return "SPRINTER" if "SPRINTER" in cands else None
        if ("E" in toks and "CITAN" in toks) or "ECITAN" in toks:
            return "CITAN" if "CITAN" in cands else None

    # Scoring
    cands = list(candidates_for_brand(marca_n, whitelist_global))
    if not cands: return None
    t_model = {tok for tok in toks if tok not in t_ruido}
    best=None; best_score=-1e9
    for cand in cands:
        s=score_candidate(cand, modelo_n, t_model, t_ruido)
        if marca_n == "MERCEDES BENZ":
            # Evita bases numéricas en Mercedes (p.ej. "220")
            if cand.isdigit():
                s -= 100
            # Si hay "CLASE X" y también existe "X" en whitelist (ej. "CLA"), favorece el código corto
            
            # Penaliza 'AMG' como base (es un acabado), salvo que no haya alternativas
            if cand == "AMG":
                s -= 500

            if cand.startswith("CLASE "):
                base = cand.split(" ", 1)[1]
                if base in set(cands):
                    s -= 5

        if s>best_score:
            best, best_score = cand, s
    return best if best_score>=50 else None

# ---------------- CLI compat + AUTODETECCIÓN ----------------

def parse_args(argv):
    p = argparse.ArgumentParser(description="Normalización de modelos (v2, compatible)")
    # Compatibilidad con el patrón antiguo y el nuevo
    p.add_argument("positional_input", nargs="?", help="(compat) input.xlsx si no usas --input")
    p.add_argument("--input", help="Ruta de entrada (xlsx/csv)")
    p.add_argument("--sheet", default=None, help="Hoja si XLSX")
    p.add_argument("--make-col", default="make", help="Columna marca (default: make)")
    p.add_argument("--model-col", default="model", help="Columna modelo (default: model)")
    p.add_argument("--whitelist", required=True, help="Ruta XLSX whitelist (Marca/Modelo)")
    p.add_argument("--output_excel", help="(compat) salida .xlsx")   # alias antiguo
    p.add_argument("--out", help="Salida .xlsx (si no usas --output_excel)")
    p.add_argument("--out-col", default=None, help="Nombre columna salida (default: modelo_base)")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    # Resolver rutas input/output según compat
    input_path = args.input or args.positional_input
    if not input_path:
        raise SystemExit("Debes indicar un input: positional o --input")

    out_path = args.output_excel or args.out
    if not out_path:
        # si no se indica salida, sobreescribir el input
        out_path = input_path

    out_col = args.out_col or "modelo_base"

    # Cargar datos
    df = _load_input(input_path, sheet=getattr(args, "sheet", None))

    # AUTODETECCIÓN de columnas si no están las inglesas
    cols_lower = {c.lower(): c for c in df.columns}
    # Marca
    if args.make_col not in df.columns:
        if "make" in cols_lower:
            args.make_col = cols_lower["make"]
        elif "marca" in cols_lower:
            args.make_col = cols_lower["marca"]
        else:
            raise SystemExit("No encuentro columna de marca (make/marca). Usa --make-col <col>")
    # Modelo
    if args.model_col not in df.columns:
        if "model" in cols_lower:
            args.model_col = cols_lower["model"]
        elif "modelo" in cols_lower:
            args.model_col = cols_lower["modelo"]
        else:
            raise SystemExit("No encuentro columna de modelo (model/modelo). Usa --model-col <col>")

    # Cargar whitelist
    wl = carga_whitelist(args.whitelist)

    # Aplicar normalización
    def _norm_row(r):
        # Atajo: si BMW y existe 'modelo_detectado' válido en whitelist, úsalo
        try:
            if normalize_brand(r[args.make_col]) == "BMW" and "modelo_detectado" in r and isinstance(r["modelo_detectado"], str):
                md = normalize_text(r["modelo_detectado"])
                if md and md in wl.get("BMW", set()):
                    return md
        except Exception:
            pass
        return normaliza_modelo(r[args.make_col], r[args.model_col], wl)

    df_out = df.copy()
    df_out[out_col] = df_out.apply(_norm_row, axis=1)

    
    # === Compatibilidad v1: columnas legadas ===
    # make_clean: marca normalizada al estilo v1
    try:
        df_out["make_clean"] = df_out[args.make_col].apply(normalize_brand)
    except Exception:
        df_out["make_clean"] = df_out[args.make_col]

    # modelo_detectado: en v2 el "mejor candidato" es el que devuelve normaliza_modelo.
    # Lo recalculamos por claridad (podría ser igual a modelo_base en v2).
    def _detect_model(r):
        try:
            return normaliza_modelo(r[args.make_col], r[args.model_col], wl)
        except Exception:
            return None
    df_out["modelo_detectado"] = df_out.apply(_detect_model, axis=1)

    # Asegurar alias en mayúsculas como en algunos v1
    if "MARCA" not in df_out.columns:
        try:
            df_out["MARCA"] = df_out[args.make_col]
        except Exception:
            pass
    if "MODELO" not in df_out.columns:
        try:
            df_out["MODELO"] = df_out[args.model_col]
        except Exception:
            pass

    # Asegurar que la columna de salida se llame 'modelo_base' como en v1 por defecto
    if out_col != "modelo_base":
        # Duplicamos para mantener compatibilidad sin romper quien pida otro nombre
        df_out["modelo_base"] = df_out[out_col]

    # Guardar salida según extensión
    p_out = Path(out_path) 
    if p_out.suffix.lower() == ".parquet":
        df_out.to_parquet(p_out, index=False)  # requiere pyarrow/fastparquet
    else:
        df_out.to_excel(p_out, index=False)

    print(f"OK -> {p_out} | filas: {len(df_out)} | columna salida: {out_col} | usando {args.make_col}/{args.model_col}")


if __name__ == "__main__":
    main()
