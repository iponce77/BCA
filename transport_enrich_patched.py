# -*- coding: utf-8 -*-
"""
Enriquecimiento de Transporte (Fase 2) — versión alineada a EN
--------------------------------------------------------------
- Canoniza países SIEMPRE a **inglés** (usa diccionario + mapa ES→EN).
- Match directo por (pais_origen_canon_en, pais_destino_canon_en) + selección por compound.
- Fallbacks: media por (pais_origen_canon_en, categoría) → media GLOBAL por categoría.
- Salidas: transport_price_eur, transport_rule, transport_confidence,
           transport_eur_fallback, transport_rule_fallback, transport_confidence_fallback,
           transport_days_estimate, y alias transport_eur.
"""

import argparse
import os
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# -------------------------
# Utilidades
# -------------------------
def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def tokenize(s: str) -> set:
    s = normalize_text(s)
    toks = re.split(r"[^0-9a-z]+", s)
    return set(t for t in toks if t)


def extract_zip_candidate(s: str) -> Optional[str]:
    m = re.search(r"\b(\d{4,6})\b", str(s or ""))
    return m.group(1) if m else None


# -------------------------
# Carga de reglas/ diccionarios
# -------------------------
def load_country_dict(path: Path) -> dict:
    """
    Espera columnas: variant -> country_canonical.
    Fallback: usa 2 primeras columnas si cambia el esquema.
    """
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    if "variant" in low and "country_canonical" in low:
        src = low["variant"]
        dst = low["country_canonical"]
    elif "raw" in low and "canon" in low:
        src = low["raw"]
        dst = low["canon"]
    else:
        cols = list(df.columns)
        if len(cols) < 2:
            raise ValueError("Diccionario de países con columnas insuficientes.")
        src, dst = cols[0], cols[1]

    d = {}
    for _, r in df.iterrows():
        d[normalize_text(r[src])] = str(r[dst]).strip()
    return d


def canon_country(name: str, cdict: dict) -> str:
    key = normalize_text(name)
    return str(cdict.get(key, name)).strip()


def _es2en_country(name: str) -> str:
    # name puede venir en ES (del country_dict) o ya en EN
    m = {
        # ES → EN
        "españa": "Spain",
        "espana": "Spain",

        "alemania": "Germany",
        "francia": "France",
        "italia": "Italy",
        "portugal": "Portugal",
        "dinamarca": "Denmark",
        "suecia": "Sweden",
        "suiza": "Switzerland",
        "austria": "Austria",
        "polonia": "Poland",
        "luxemburgo": "Luxembourg",
        "paises bajos": "Netherlands",
        "países bajos": "Netherlands",
        "bélgica": "Belgium",
        "belgica": "Belgium",
        "finlandia": "Finland",
        "europa": "Europe",
    }

    key = normalize_text(name)

    # Sinónimos en inglés/europeos → Europe
    if key in {"europe", "eu", "ue", "european union", "union europea", "union europeenne"}:
        return "Europe"

    # Si es un canónico ES conocido, lo traduzco;
    # si ya está en EN o es algo raro, lo devuelvo tal cual.
    return m.get(key, name)



def canon_country_en(name: str, cdict: dict) -> str:
    # 1) pasa por el diccionario (puede devolver español)
    v = canon_country(name, cdict)
    # 2) fuerza canónico en inglés
    return _es2en_country(v)


def load_json_list(path: Optional[Path]) -> List[str]:
    if not path or not path.exists():
        return []
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict) and "tokens" in data:
        return list(data["tokens"])
    if isinstance(data, list):
        return [str(x) for x in data]
    return []


def load_body_map(path: Optional[Path]) -> dict:
    """Carga JSON con reglas de mapeo de carrocería → categoría.
    Si falta o no es válido, devuelve {}.
    Estructura esperada:
      {
        "passenger_car": ["berlina","compacto", ...],
        "suv": ["suv","estate", ...],
        "lcv": ["van","furgon", ...],
        "patterns": [{"pattern": "regex", "category":"suv"}, ...]
      }
    """
    if not path or not path.exists():
        return {}
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}

def load_location_aliases(path: Optional[Path], cdict: Optional[dict] = None) -> dict:
    """Lee CSV con columnas: location_head, country_canonical → dict normalizado por head.
       Convierte country_canonical a inglés para alinear con rutas."""
    if not path or not path.exists():
        return {}
    try:
        import pandas as _pd
        df = _pd.read_csv(path)
    except Exception:
        return {}
    cols = {c.lower(): c for c in df.columns}
    head_col = cols.get("location_head") or list(df.columns)[0]
    country_col = cols.get("country_canonical") or (list(df.columns)[1] if len(df.columns) > 1 else head_col)
    out = {}
    for _, r in df.iterrows():
        head = normalize_text(str(r[head_col]))
        out[head] = canon_country_en(r[country_col], cdict or {})
    return out


# -------------------------
# Mapeo de categoría
# -------------------------
CATEGORIES = ("passenger_car", "suv", "lcv")


def map_category(row: pd.Series, body_map: dict) -> str:
    # Fuentes: Tipo Carrocería → Tipo de vehículo → vehicle_type
    candidates = [
        row.get("Tipo Carrocería", ""),
        row.get("Tipo de vehículo", ""),
        row.get("vehicle_type", ""),
    ]
    txt = " ".join([str(x) for x in candidates if pd.notna(x)])

    # 1) Reglas declarativas (tokens por categoría)
    for cat in CATEGORIES:
        for token in body_map.get(cat, []):
            if token and normalize_text(token) in normalize_text(txt):
                return cat

    # 2) Patrones regex opcionales
    for pat in body_map.get("patterns", []):
        patt = pat.get("pattern", "")
        cat = pat.get("category", "")
        try:
            if cat in CATEGORIES and patt and re.search(patt, txt, re.IGNORECASE):
                return cat
        except re.error:
            continue

    # 3) Heurística fallback
    n = normalize_text(txt)
    if any(k in n for k in ["suv", "estate", "station", "touring", "allroad"]):
        return "suv"
    if any(k in n for k in ["van", "furg", "lcv", "box", "chasis", "chassis", "pickup", "transporter"]):
        return "lcv"
    return "passenger_car"


def pick_price_col(cat: str, df_trans: pd.DataFrame) -> str:
    if cat == "suv":
        return "precio_suv_eur"
    if cat == "lcv":
        if "precio_lcv2_eur" in df_trans.columns:
            return "precio_lcv2_eur"
        if "precio_lcv1_eur" in df_trans.columns:
            return "precio_lcv1_eur"
    return "precio_passenger_car_eur"


# -------------------------
# Scoring de compound
# -------------------------
def compound_score(ubicacion: str,
                   origen_compound: str,
                   brand_tokens: set,
                   stop_tokens: set,
                   destino_zip: Optional[str]) -> float:
    u_tokens = tokenize(ubicacion)
    c_tokens = tokenize(origen_compound)

    if stop_tokens:
        c_tokens = {t for t in c_tokens if t not in stop_tokens}

    overlap = len(u_tokens & c_tokens)
    brand_hits = len({t for t in u_tokens if t in brand_tokens})

    score = overlap * 1.0 + brand_hits * 1.5

    # Bonus ZIP
    zip_in_ubi = extract_zip_candidate(ubicacion)
    if zip_in_ubi and destino_zip and str(zip_in_ubi) == str(destino_zip):
        score += 2.0

    # Bonus de arranque similar
    if origen_compound and ubicacion and normalize_text(ubicacion).startswith(normalize_text(origen_compound)[:10]):
        score += 0.5

    return score


def select_transport_row(df_routes: pd.DataFrame,
                         ubicacion: str,
                         brand_list: List[str],
                         stop_list: List[str]) -> Optional[pd.Series]:
    if df_routes.empty:
        return None

    brand_tokens = {normalize_text(x) for x in brand_list}
    stop_tokens = {normalize_text(x) for x in stop_list}

    scores = []
    for _, r in df_routes.iterrows():
        s = compound_score(
            ubicacion=str(ubicacion or ""),
            origen_compound=str(r.get("origen_compound", "")),
            brand_tokens=brand_tokens,
            stop_tokens=stop_tokens,
            destino_zip=r.get("destino_zip"),
        )
        scores.append(s)

    df_routes = df_routes.copy()
    df_routes["__score__"] = scores
    df_routes = df_routes.sort_values(["__score__"], ascending=False)
    return df_routes.iloc[0]


# -------------------------
# Fallbacks de media
# -------------------------
def compute_country_category_means(df_trans: pd.DataFrame):
    rows = []
    # coalesce LCV
    df_lcv = df_trans.copy()
    if "precio_lcv2_eur" in df_lcv.columns or "precio_lcv1_eur" in df_lcv.columns:
        df_lcv["__precio_lcv__"] = df_lcv["precio_lcv2_eur"].where(
            pd.notna(df_lcv.get("precio_lcv2_eur")), df_lcv.get("precio_lcv1_eur")
        )

    map_cols = [
        ("passenger_car", "precio_passenger_car_eur"),
        ("suv", "precio_suv_eur"),
        ("lcv", "__precio_lcv__"),  # ← usar coalesce
    ]
    for cat, col in map_cols:
        if not col or col not in df_lcv.columns:
            continue
        tmp = df_lcv[["pais_origen_canon", col]].copy()
        tmp = tmp.rename(columns={"pais_origen_canon": "pais_origen", col: "price"})
        tmp["category"] = cat
        rows.append(tmp)

    base = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(columns=["pais_origen", "price", "category"])
    base = base[pd.to_numeric(base["price"], errors="coerce").notnull()]
    base["price"] = base["price"].astype(float)

    df_means_country = (
        base.groupby(["pais_origen", "category"], dropna=False)["price"]
        .mean().reset_index().rename(columns={"price": "mean_price"})
    )
    df_means_global = (
        base.groupby(["category"], dropna=False)["price"]
        .mean().reset_index().rename(columns={"price": "mean_price"})
    )
    return df_means_country, df_means_global


# -------------------------
# Enriquecimiento por fila
# -------------------------
def enrich_row(row: pd.Series,
               df_trans: pd.DataFrame,
               df_means_country: pd.DataFrame,
               df_means_global: pd.DataFrame,
               country_dict: dict,
               body_map: dict,
               dest_canon: str,
               brand_list: List[str],
               stop_list: List[str],
               location_aliases: dict) -> pd.Series:

    # 1) origen/destino canon EN
    origin = row.get("sale_info_country") or row.get("saleCountry") or row.get("sale_country") or ""
    origin_canon = canon_country_en(str(origin), country_dict)
    ubicacion = str(row.get("Ubicación", "") or row.get("Ubicacion", "") or "")

    # si el país de origen es genérico (EU/Europa/vacío), inferir desde Ubicación → alias ya en EN
    if normalize_text(origin_canon) in {"eu","ue","europe","europa","","nan","none","global"}:
        head = ubicacion.split(" - ")[0].split(",")[0].strip() if ubicacion else ""
        if head:
            key = normalize_text(head)
            mapped = location_aliases.get(key)
            if mapped:
                origin_canon = mapped
            else:
                origin_canon = canon_country_en(head, country_dict)

    # 2) categoría y columna de precio
    cat = map_category(row, body_map)
    price_col = pick_price_col(cat, df_trans)

    # 3) candidatos directos por país (normalizado)
    mask = (
        (df_trans["pais_origen_canon"].astype(str).map(normalize_text) == normalize_text(origin_canon)) &
        (df_trans["pais_destino_canon"].astype(str).map(normalize_text) == normalize_text(dest_canon))
    )
    cand = df_trans.loc[mask].copy()

    transport_price = None
    rule = None
    conf = None
    days = None

    # 4) selección por compound
    if not cand.empty and price_col in cand.columns:
        best = select_transport_row(cand, ubicacion, brand_list, stop_list)
        if best is not None:
            used_col = None
            if cat == "lcv":
                v2 = best.get("precio_lcv2_eur", np.nan)
                v1 = best.get("precio_lcv1_eur", np.nan)
                val = v2 if pd.notna(v2) else v1
                used_col = "precio_lcv2_eur" if pd.notna(v2) else ("precio_lcv1_eur" if pd.notna(v1) else None)
            else:
                col = pick_price_col(cat, df_trans)  # mantiene passenger/suv
                val = best.get(col, np.nan)
                used_col = col 
                   
            if pd.notna(val):
                transport_price = float(val)
                col_tag = used_col if used_col else "n/a"
                rule = f"direct|compound={best.get('origen_compound','')}|score={best.get('__score__',0):.2f}|cat={cat}|col={col_tag}"
                conf = 1.0
                for dc in ["dias_transporte", "dias_single_cars", "dias_full_truck", "dias_est", "dias"]:
                    if dc in best and pd.notna(best[dc]):
                        try:
                            days = int(float(best[dc]))
                            break
                        except Exception:
                            continue

    # 5) fallbacks si no hay precio directo
    fb_price = None
    fb_rule = None
    fb_conf = None

    if transport_price is None:
        row_mean = df_means_country[
            (df_means_country["pais_origen"].astype(str).map(normalize_text) == normalize_text(origin_canon)) &
            (df_means_country["category"] == cat)
        ]
        if not row_mean.empty:
            fb_price = float(row_mean["mean_price"].values[0])
            fb_rule = f"fallback_country|origin={origin_canon}|cat={cat}"
            fb_conf = 0.5
        else:
            row_g = df_means_global[df_means_global["category"] == cat]
            if not row_g.empty:
                fb_price = float(row_g["mean_price"].values[0])
                fb_rule = f"fallback_global|cat={cat}"
                fb_conf = 0.3

    # 6) salida
    final_price = transport_price if transport_price is not None else (fb_price if fb_price is not None else 0.0)
    final_rule = rule if rule is not None else (fb_rule if fb_rule is not None else "sin_tarifa_transporte")
    final_conf = conf if conf is not None else (fb_conf if fb_conf is not None else 0.0)

    row_out = row.copy()
    row_out["transport_price_eur"] = final_price
    row_out["transport_rule"] = final_rule
    row_out["transport_confidence"] = final_conf
    row_out["transport_eur"] = final_price
    row_out["transport_eur_fallback"] = fb_price if fb_price is not None else np.nan
    row_out["transport_rule_fallback"] = fb_rule if fb_rule is not None else ""
    row_out["transport_confidence_fallback"] = fb_conf if fb_conf is not None else np.nan
    row_out["transport_days_estimate"] = days if days is not None else np.nan

    return row_out


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Enriquecimiento de Transporte (Fase 2)")
    ap.add_argument("--excel", required=True, help="Excel de fichas de entrada (por ejemplo, fichas_vehiculos_20250619.xlsx)")
    ap.add_argument("--transporte", required=True, help="CSV estructurado de transporte (por ejemplo, bca_transporte_estructurado.csv)")
    ap.add_argument("--country_dict", required=True, help="CSV diccionario de países (variant → country_canonical)")
    ap.add_argument("--body_rules", required=False, default="", help="JSON de mapeo carrocería→categoría")
    ap.add_argument("--brands", required=False, default="", help="JSON de tokens de marcas/operadores (opcional)")
    ap.add_argument("--stops", required=False, default="", help="JSON de tokens a ignorar en compound (opcional)")
    ap.add_argument("--dest", required=False, default="", help="País de destino (por ejemplo, 'Spain') o usar env DEST_COUNTRY")
    ap.add_argument("--location_aliases", required=False, default="", help="CSV de alias de location_head -> país canónico")
    ap.add_argument("--sheet", required=False, default=None, help="Nombre de hoja del Excel si aplica")
    ap.add_argument("--out", required=False, default="", help="Excel de salida enriquecido (por defecto, sobrescribe --excel)")
    args = ap.parse_args()

    # Derivar DEST si no viene como flag
    dest_env = os.environ.get("DEST_COUNTRY", "").strip()
    if not args.dest:
        if dest_env:
            args.dest = dest_env
        else:
            raise SystemExit("ERROR: falta --dest y no hay DEST_COUNTRY en entorno")

    # Derivar OUT si no se pasó: sobrescribe el excel de entrada
    if not args.out:
        args.out = args.excel

    excel_path = Path(args.excel)
    trans_path = Path(args.transporte)
    country_path = Path(args.country_dict)
    body_path = Path(args.body_rules) if args.body_rules else None
    brands_path = Path(args.brands) if args.brands else None
    stops_path = Path(args.stops) if args.stops else None
    loc_aliases_path = Path(args.location_aliases) if args.location_aliases else None
    out_path = Path(args.out)

    # Carga datos
    df_fichas = pd.read_excel(excel_path, sheet_name=args.sheet) if args.sheet else pd.read_excel(excel_path)
    df_trans = pd.read_csv(trans_path)

    # Carga reglas/listas
    country_dict = load_country_dict(country_path)
    body_map = load_body_map(body_path) if body_path else {}
    brand_list = load_json_list(brands_path)
    stop_list = load_json_list(stops_path)
    location_aliases = load_location_aliases(loc_aliases_path, country_dict)

    # Canoniza países también en el CSV de transporte **a EN**
    for col in ["pais_origen", "pais_destino"]:
        if col in df_trans.columns:
            df_trans[f"{col}_canon"] = df_trans[col].apply(lambda x: canon_country_en(x, country_dict))
        else:
            df_trans[f"{col}_canon"] = np.nan

    # Destino canon (EN)
    dest_canon = canon_country_en(args.dest, country_dict)

    # Precalcular medias (por pais_origen_canon EN, categoría)
    df_means_country, df_means_global = compute_country_category_means(df_trans)

    # Enriquecer
    enriched = df_fichas.apply(
        enrich_row,
        axis=1,
        df_trans=df_trans,
        df_means_country=df_means_country,
        df_means_global=df_means_global,
        country_dict=country_dict,
        body_map=body_map,
        dest_canon=dest_canon,
        brand_list=brand_list,
        stop_list=stop_list,
        location_aliases=location_aliases,
    )

    # Guardar
    enriched.to_excel(out_path, index=False, engine="openpyxl")
    print(f"[OK] Transporte enriquecido → {out_path}")


if __name__ == "__main__":
    main()
