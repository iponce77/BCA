
# bca_enrich_lib.py
# -*- coding: utf-8 -*-
"""
Librería unificada para el pipeline de enriquecimiento BCA (matching strict→relax + enriquecimiento + análisis).
Se inspira y reutiliza la lógica existente en:
- merge_master_isaac_strict_relax.py (filtros ordenados con relajaciones, ranking por tokens, auditoría top‑K)  # ver scripts originales
- enrich_bca_with_master.py (normalización de row_id_bca, ROI y hojas de análisis)                               # ver scripts originales

Notas clave del diseño:
- Esta librería **no escribe a disco**. El I/O queda a cargo del CLI (bca_enrich_all.py).
- Firmas públicas exigidas:
    * load_config(path: str) -> dict
    * run_matching(bca_path: str, master_path: str, rules: dict, out_topk: str | None) -> (pd.DataFrame, pd.DataFrame)
      Devuelve (matches, audit_topk). El parámetro out_topk se acepta por compatibilidad pero **no** se usa para escribir.
    * run_enrichment(bca_path: str, matches: pd.DataFrame, master_path: str, params: dict) -> (pd.DataFrame, dict[str, pd.DataFrame])
      Devuelve (bca_enriched, {nombre_hoja: df}).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher as _SequenceMatcher

# ----------------- Logging -----------------

def setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------- Text utils -----------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[^\w\s]", " ", s)  # letters/digits/underscore/space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: Any, stopwords: Set[str]) -> List[str]:
    t = normalize_text(s)
    if not t:
        return []
    toks = [tok for tok in t.split() if tok not in stopwords]
    return toks

def token_overlap(a: List[str], b: List[str]) -> Tuple[int, List[str]]:
    sa, sb = set(a), set(b)
    inter = sorted(sa.intersection(sb))
    return len(inter), inter

def fuzzy_ratio(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if _HAS_RAPIDFUZZ:
        ts = _rf_fuzz.token_set_ratio(a, b, score_cutoff=0)
        pr = _rf_fuzz.partial_ratio(a, b, score_cutoff=0)
        return int(max(ts, pr))
    return int(100 * _SequenceMatcher(None, a, b).ratio())

def normalize_headers(cols: List[str]) -> List[str]:
    return [c.replace("\xa0", " ").strip() for c in cols]

# ----------------- Archivo / Config -----------------

def read_any(path: str) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    elif p.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    elif p.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        # intento xlsx por defecto
        return pd.read_excel(path)

def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class ColumnMap:
    required_map: Dict[str, List[str]]
    def resolve(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        cols = list(df.columns)
        cols_norm = {c.lower(): c for c in cols}
        out: Dict[str, Optional[str]] = {}
        for internal, cand in self.required_map.items():
            found: Optional[str] = None
            for name in cand:
                # exact (case-insensitive)
                if name.lower() in cols_norm:
                    found = cols_norm[name.lower()]
                    break
                # contains (case-insensitive)
                matches = [c for c in cols if name.lower() in c.lower()]
                if matches:
                    found = matches[0]; break
            out[internal] = found
        return out

# ----------------- Canonicalización -----------------

def canonicalize_fuel(v: Any, aliases: Dict[str, str]) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, pd.Series) and v.isna().all()):
        return None
    # normalizar claves del mapa
    m = {normalize_text(k).upper(): val for k, val in (aliases or {}).items()}
    m.update({normalize_text(val).upper(): val for val in (aliases or {}).values()})
    key = normalize_text(v).upper()
    return m.get(key, str(v).upper()) if key else None

# ----------------- Prepare BCA / Ganvam -----------------

def prepare_bca(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df.columns = normalize_headers(list(df.columns))
    cmap = ColumnMap(cfg["columns"]["bca"]).resolve(df)

    out = pd.DataFrame(index=df.index)
    out["row_id_bca"] = df.index.astype(str)

    # Marca / modelo_base / model raw
    out["marca"] = df[cmap["marca"]] if cmap.get("marca") else df.get("make_clean", df.get("make",""))
    out["modelo_base"] = df[cmap["modelo_base"]] if cmap.get("modelo_base") else df.get("modelo_base", df.get("modelo_base_match",""))
    out["model_bca_raw"] = df[cmap["model_bca_raw"]] if cmap.get("model_bca_raw") else df.get("model","")

    # Año
    year_col = cmap.get("year_bca")
    if year_col and year_col in df.columns:
        y1 = pd.to_numeric(df[year_col], errors="coerce")
        y2 = pd.to_datetime(df[year_col], errors="coerce", dayfirst=True).dt.year
        out["year_bca"] = y1.fillna(y2).astype("Int64")
    else:
        out["year_bca"] = pd.Series(pd.array([pd.NA]*len(df), dtype="Int64"))

    # CV y combustible
    out["cv_bca"] = pd.to_numeric(df[cmap["cv_bca"]], errors="coerce") if cmap.get("cv_bca") else pd.to_numeric(df.get("cv"), errors="coerce")
    out["combustible_bca"] = df[cmap["combustible_bca"]] if cmap.get("combustible_bca") else df.get("fuel_type")

    # Claves normalizadas
    out["marca_key"] = out["marca"].map(normalize_text)
    out["modelo_base_key"] = out["modelo_base"].map(normalize_text)
    return out.reset_index(drop=True)

def prepare_ganvam(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df.columns = normalize_headers(list(df.columns))
    cmap = ColumnMap(cfg["columns"]["ganvam"]).resolve(df)

    out = pd.DataFrame(index=df.index)
    out["row_id_ganvam"] = df.index.astype(str)
    out["marca"] = df[cmap["marca"]] if cmap.get("marca") else df.get("make_clean", df.get("make",""))
    out["modelo_base"] = df[cmap["modelo_base"]] if cmap.get("modelo_base") else df.get("modelo_base","")
    out["model_ganvam_raw"] = df[cmap["model_ganvam_raw"]] if cmap.get("model_ganvam_raw") else df.get("model_raw", df.get("model",""))
    out["version_ganvam"] = df[cmap["version_ganvam"]] if cmap.get("version_ganvam") else df.get("version")
    out["precio_venta_ganvam"] = pd.to_numeric(df[cmap["precio_venta_ganvam"]], errors="coerce") if cmap.get("precio_venta_ganvam") else pd.to_numeric(df.get("precioVenta"), errors="coerce")
    out["anio_ganvam"] = pd.to_numeric(df[cmap["anio_ganvam"]], errors="coerce") if cmap.get("anio_ganvam") else pd.to_numeric(df.get("anio"), errors="coerce")
    out["cv_ganvam"] = pd.to_numeric(df[cmap["cv_ganvam"]], errors="coerce") if cmap.get("cv_ganvam") else pd.to_numeric(df.get("cv"), errors="coerce")
    out["combustible_ganvam"] = df[cmap["combustible_ganvam"]] if cmap.get("combustible_ganvam") else df.get("combustible_desc")
    out["startYear"] = pd.to_numeric(df.get("startYear"), errors="coerce")
    out["endYear"] = pd.to_numeric(df.get("endYear"), errors="coerce")
    out["marca_key"] = out["marca"].map(normalize_text)
    out["modelo_base_key"] = out["modelo_base"].map(normalize_text)
    return out.reset_index(drop=True)

# ----------------- Filtros strict + relax -----------------
def year_filter(base: pd.DataFrame, year_bca: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    trace: Dict[str, Any] = {}
    if pd.isna(year_bca):
        trace["applied"] = False
        trace["dropped"] = True
        return base, trace

    yb = int(year_bca)
    trace["applied"] = True

    # 1) Filtro ESTRICTO por igualdad exacta
    eq = base[base["anio_ganvam"].astype("Int64") == yb]
    trace["n_strict"] = int(len(eq))
    if len(eq) > 0:
        return eq, trace

    # 2) Filtro RELAJADO por rango SOLO si el año BCA es anterior a 2013
    if yb < 2013 and ("startYear" in base.columns or "endYear" in base.columns):
        start_col = base.get("startYear", pd.Series(np.nan, index=base.index)).fillna(yb)
        end_col   = base.get("endYear",   pd.Series(np.nan, index=base.index)).fillna(yb)
        in_range = (yb >= start_col) & (yb <= end_col)
        eq = base[in_range]
        trace["n_range"] = int(len(eq))
        trace["relaxed_year_range"] = True if len(eq) > 0 else False
        if len(eq) > 0:
            return eq, trace

    # 3) Si no hay candidatos, marcamos caída por año
    trace["dropped"] = True
    return base, trace

def fuel_filter_with_tolerance(base: pd.DataFrame, bca_fuel: Any, aliases: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    VERSIÓN DURA: ya NO aplica tolerancias. Canonicaliza ambos lados y exige igualdad exacta.
    """
    trace: Dict[str, Any] = {}
    bf = canonicalize_fuel(bca_fuel, aliases)  # usa tus fuel_aliases*
    if not bf or str(bf).strip() == "":
        trace["applied"] = False
        trace["dropped"] = True
        trace["reason"] = "fuel_bca_missing"
        return base, trace

    # Detecta columna canónica de GANVAM
    col_candidates = ["combustible_ganvam_canon", "fuel_ganvam_canon", "combustible_canon", "canonical_fuel", "combustible_ganvam"]
    col = next((c for c in col_candidates if c in base.columns), None)
    if col is None:
        trace["applied"] = True
        trace["dropped"] = True
        trace["reason"] = "ganvam_fuel_column_missing"
        return base, trace

    # Canonicaliza GANVAM si aún no está canónico
    if col != "combustible_ganvam_canon":
        gf = base[col].map(lambda v: canonicalize_fuel(v, aliases))
    else:
        gf = base[col].astype(str).str.lower()

    # Filtro duro: igualdad exacta
    strict_mask = gf.astype(str).str.lower().eq(str(bf).lower())
    strict = base[strict_mask]

    trace["applied"] = True
    trace["n_strict"] = int(len(strict))
    if len(strict) > 0:
        return strict, trace

    # Nada de relajaciones:
    trace["dropped"] = True
    trace["reason"] = f"fuel_mismatch_hard({bf})"
    return base, trace

def cv_filter_with_relax(base: pd.DataFrame, cv_bca: Any, allow_kw_cv_equivalence: bool=False) -> Tuple[pd.DataFrame, Dict[str,Any]]:
    trace: Dict[str,Any] = {}
    if pd.isna(cv_bca):
        trace["applied"] = False
        trace["dropped"] = True
        return base, trace
    gb = base["cv_ganvam"]
    eq = (gb.astype("Int64") == int(cv_bca))
    if allow_kw_cv_equivalence:
        # equivalencia aproximada kW↔CV (1 CV ≈ 0.7355 kW) => 1 kW ≈ 1.35962 CV
        conv = int(round(float(cv_bca) * 1.35962))
        eq = eq | (gb.astype("Int64") == conv)
    strict = base[eq]
    trace["applied"] = True
    trace["n_strict"] = int(len(strict))
    if len(strict) > 0:
        return strict, trace
    trace["dropped"] = True  # se usará como tie-breaker más tarde
    return base, trace

# ----------------- Fuzzy blocking por modelo_base -----------------

def fuzzy_model_candidates(base: pd.DataFrame, model_key: str, threshold: float = 90.0, topk: int = 3) -> pd.DataFrame:
    if base.empty or not model_key:
        return base
    if not _HAS_RAPIDFUZZ:
        return base
    scores: List[Tuple[int, float]] = []
    for idx, row in base.iterrows():
        cand_key = row.get("modelo_base_key", "")
        if not cand_key:
            continue
        score = _rf_fuzz.token_set_ratio(model_key, cand_key)
        scores.append((idx, score))
    selected = [i for i, sc in sorted(scores, key=lambda x: x[1], reverse=True) if sc >= threshold]
    if not selected:
        return base
    selected = selected[:topk]
    return base.loc[selected]

# ----------------- Ranking -----------------

def build_text_series(base: pd.DataFrame, cfg: Dict[str,Any], stopwords: Set[str]) -> pd.Series:
    src = cfg.get("ranking", {}).get("ganvam_text_source", "version_only")
    if src not in ("version_only", "model_plus_version"):
        src = "version_only"
    if src == "version_only":
        return (base["version_ganvam"].fillna("")).map(lambda s: " ".join(tokenize(s, stopwords)))
    return (base["model_ganvam_raw"].fillna("") + " " + base["version_ganvam"].fillna("")).map(lambda s: " ".join(tokenize(s, stopwords)))

def score_by_tokens(b_txt: str, cand_text: pd.Series) -> pd.DataFrame:
    b_tok = b_txt.split()
    inter_sizes = []
    inter_lists = []
    substr = []
    for s in cand_text.tolist():
        toks = s.split()
        cnt, inter = token_overlap(b_tok, toks)
        inter_sizes.append(cnt)
        inter_lists.append(" ".join(inter))
        s_norm = " ".join(toks)
        b_norm = " ".join(b_tok)
        substr.append(int(b_norm in s_norm or s_norm in b_norm))
    return pd.DataFrame({"token_overlap": inter_sizes, "token_overlap_terms": inter_lists, "substring_hit": substr})

# ----------------- Selección ordenada con relax -----------------

def select_with_order_and_relax(base: pd.DataFrame, b_row: pd.Series, cfg: Dict[str,Any], stage_label:str) -> Tuple[Optional[pd.Series], pd.DataFrame, Dict[str,Any]]:
    trace = {
        'stage': stage_label,
        'n_start': int(len(base)),
        'year': {},
        'fuel': {},
        'cv': {},
        'n_after_year': None,
        'n_after_fuel': None,
        'n_after_cv': None,
        'n_after_tokens': None,
        'cv_tiebreak_used': False,
        'fail_step': None
    }
    if base.empty:
        trace['fail_step'] = 'no_candidates_start'
        return None, base, trace

    base0 = base.copy()

    # 1) YEAR
    base_y, ytr = year_filter(base, b_row.get("year_bca"))
    trace['year'] = ytr
    trace['n_after_year'] = int(len(base_y))
    if base_y.empty:
        base_y = base0.copy()

    # 2) FUEL
    base_yf0 = base_y.copy()
    base_f, ftr = fuel_filter_with_tolerance(base_y, b_row.get("combustible_bca"), cfg.get("aliases",{}).get("fuel",{}))
    trace['fuel'] = ftr
    trace['n_after_fuel'] = int(len(base_f))
    if base_f.empty:
        base_f = base_yf0.copy()

    # 3) CV (con relax)
    base_fc0 = base_f.copy()
    base_c, ctr = cv_filter_with_relax(base_f, b_row.get("cv_bca"), allow_kw_cv_equivalence=bool(cfg.get("filters",{}).get("allow_kw_cv_equivalence", False)))
    trace['cv'] = ctr
    trace['n_after_cv'] = int(len(base_c))
    if base_c.empty:
        base_c = base_fc0.copy()

    if base_c.empty:
        trace['fail_step'] = 'no_candidates_after_filters'
        return None, base_c, trace

    # 4) Tokens
    stopwords = set([normalize_text(w) for w in cfg.get("model_text", {}).get("stopwords", [])])
    b_text = " ".join(tokenize(b_row.get("model_bca_raw"), stopwords))
    cand_text = build_text_series(base_c, cfg, stopwords)
    scores = score_by_tokens(b_text, cand_text)
    base_c = base_c.copy().reset_index(drop=True)
    base_c = pd.concat([base_c, scores], axis=1)

    min_tok = int(cfg.get("ranking", {}).get("min_common_tokens", 1))
    base_c = base_c[base_c["token_overlap"] >= min_tok]
    trace['n_after_tokens'] = int(len(base_c))
    if base_c.empty:
        trace['fail_step'] = 'token_min'
        return None, base_c, trace

    # Ranking por tokens + fuzzy backup
    base_c["fuzzy_backup"] = base_c["version_ganvam"].fillna("").map(lambda t: fuzzy_ratio(b_text, normalize_text(t)) / 100.0)
    base_c = base_c.sort_values(by=["token_overlap", "substring_hit", "fuzzy_backup"], ascending=[False, False, False]).copy()

    # Tie-breaker CV si se cayó y BCA tiene CV
    cv_bca = b_row.get("cv_bca")
    topk = int(cfg.get("ranking",{}).get("topk_for_cv_tiebreaker", 3))
    if (trace['cv'].get('dropped', False) is True) and pd.notna(cv_bca):
        head = base_c.head(topk).copy()
        if not head.empty:
            head["cv_delta_abs"] = (head["cv_ganvam"] - float(cv_bca)).abs()
            head = head.sort_values(by=["cv_delta_abs","token_overlap","substring_hit","fuzzy_backup"], ascending=[True, False, False, False])
            picked = head.iloc[0].copy()
            trace['cv_tiebreak_used'] = True
            base_c = pd.concat([head, base_c.iloc[topk:]], ignore_index=True)
        else:
            picked = base_c.iloc[0].copy()
    else:
        picked = base_c.iloc[0].copy()

    picked["matching_type"] = "KEY_STRICT" if stage_label=="EXACT" else ("APROX_SUBSTRING" if int(picked["substring_hit"])==1 else "APROX_TOKENS")
    picked["is_approximate"] = stage_label!="EXACT"
    return picked, base_c, trace

# ----------------- Matching pipeline -----------------

def run_matching(bca_path: str, master_path: str, rules: Dict[str, Any], out_topk: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el matching strict→relax y devuelve:
      - matches: DataFrame con una fila por BCA (cuando hay match) y las columnas de diagnóstico pedidas.
      - audit_topk: DataFrame con los candidatos Top‑K por fila (para auditoría). Incluye filas "unmatched" con diagnóstico de fallos.
    NOTA: Este método **no escribe archivos** aunque reciba out_topk (compatibilidad).
    """
    # Cargar entradas
    bca_raw = read_any(bca_path)
    ganvam_raw = read_any(master_path)

    # Preparación
    bca = prepare_bca(bca_raw, rules)
    ganvam = prepare_ganvam(ganvam_raw, rules)

    out_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []

    # Hints manuales (opcional) - se pueden añadir en el futuro vía rules["hints_path"]
    hints_df = None
    hint_map: Dict[str, str] = {}
    if rules.get("inputs", {}).get("hints"):
        try:
            hints_df = read_any(rules["inputs"]["hints"])
        except Exception as e:
            logging.warning("No se pudo leer hints: %s", e)
    if hints_df is not None and not hints_df.empty:
        bca_col = next((c for c in ["row_id_bca","ROW_ID_BCA","id_bca","ID_BCA"] if c in hints_df.columns), None)
        gcol = next((c for c in ["row_id_ganvam","ROW_ID_GANVAM","FILA CORRECTA","FILA_CORRECTA","fila_correcta"] if c in hints_df.columns), None)
        if bca_col and gcol:
            for _, rr in hints_df[[bca_col,gcol]].dropna().iterrows():
                try:
                    b_id = str(rr[bca_col])
                    g_id = int(float(rr[gcol]))  # podría venir "123.0"
                    hint_map[b_id] = str(g_id)
                except Exception:
                    continue

    # Iteración por filas BCA
    for _, r in bca.iterrows():
        # Stage 0: override por hint
        if r["row_id_bca"] in hint_map:
            gid = hint_map[r["row_id_bca"]]
            gmatch = ganvam[ganvam["row_id_ganvam"] == gid]
            if not gmatch.empty:
                g = gmatch.iloc[0]
                out_rows.append({
                    "row_id_bca": r["row_id_bca"],
                    "marca": r["marca"],
                    "modelo_base": r["modelo_base"],
                    "model_bca_raw": r["model_bca_raw"],
                    "year_bca": r["year_bca"],
                    "cv_bca": r["cv_bca"],
                    "combustible_bca": r.get("combustible_bca"),
                    "row_id_ganvam": g.get("row_id_ganvam"),
                    "model_ganvam_raw": g.get("model_ganvam_raw"),
                    "version_ganvam": g.get("version_ganvam"),
                    "anio_ganvam": g.get("anio_ganvam"),
                    "cv_ganvam": g.get("cv_ganvam"),
                    "combustible_ganvam": g.get("combustible_ganvam"),
                    "precio_venta_ganvam": g.get("precio_venta_ganvam"),
                    "matching_type": "HINT_OVERRIDE",
                    "is_approximate": False,
                    "token_overlap": np.nan,
                    "token_overlap_terms": "",
                    "substring_hit": np.nan,
                    "stage": "HINT",
                    "year_dropped": False,
                    "fuel_relaxed_evhybrid": False,
                    "fuel_dropped": False,
                    "cv_dropped": False,
                })
                audit_rows.append({"row_id_bca": r["row_id_bca"], "cand_rank": 0, "row_id_ganvam": gid, "decision": "selected_by_hint"})
                continue

        # Stage A: EXACT por marca+modelo_base
        exact = ganvam[(ganvam["marca_key"] == r["marca_key"]) & (ganvam["modelo_base_key"] == r["modelo_base_key"])].copy()
        picked = None
        cand_df = None
        stage = None
        exact_trace = None
        approx_trace = None

        if not exact.empty:
            sel, cand, trace = select_with_order_and_relax(exact, r, rules, stage_label="EXACT")
            exact_trace = trace
            if sel is not None:
                picked = sel
                cand_df = cand
                stage = "EXACT"

        # Stage B: por marca + fuzzy blocking opcional
        if picked is None:
            brand = ganvam[ganvam["marca_key"] == r["marca_key"]].copy()
            brand_candidates = brand
            use_fuzzy = bool(rules.get("filters", {}).get("use_fuzzy_model_base", False))
            if use_fuzzy and not brand.empty:
                fz_threshold = float(rules.get("filters", {}).get("fuzzy_model_base_threshold", 90.0))
                fz_topk = int(rules.get("filters", {}).get("topk_modelo_keys", 3))
                try:
                    fuzzy_subset = fuzzy_model_candidates(brand, r.get("modelo_base_key", ""), threshold=fz_threshold, topk=fz_topk)
                except Exception:
                    fuzzy_subset = brand
                if not fuzzy_subset.empty and len(fuzzy_subset) < len(brand):
                    brand_candidates = fuzzy_subset.copy()
            sel, cand, trace = select_with_order_and_relax(brand_candidates, r, rules, stage_label="APPROX")
            approx_trace = trace
            if sel is not None:
                picked = sel
                cand_df = cand
                stage = "APPROX"

        # Salidas
        if picked is None:
            uf = {
                "row_id_bca": r["row_id_bca"],
                "marca": r["marca"],
                "modelo_base": r["modelo_base"],
                "model_bca_raw": r["model_bca_raw"],
                "year_bca": r.get("year_bca"),
                "cv_bca": r.get("cv_bca"),
                "combustible_bca": r.get("combustible_bca"),
                "marca_key": r["marca_key"],
                "modelo_base_key": r["modelo_base_key"],
                "exact_fail_step": (exact_trace or {}).get("fail_step", "no_exact_candidates" if exact.empty else "unknown"),
                "approx_fail_step": (approx_trace or {}).get("fail_step", "unknown"),
                "exact_n_start": (exact_trace or {}).get("n_start"),
                "exact_n_after_year": (exact_trace or {}).get("n_after_year"),
                "exact_n_after_fuel": (exact_trace or {}).get("n_after_fuel"),
                "exact_n_after_cv": (exact_trace or {}).get("n_after_cv"),
                "exact_n_after_tokens": (exact_trace or {}).get("n_after_tokens"),
                "approx_n_start": (approx_trace or {}).get("n_start"),
                "approx_n_after_year": (approx_trace or {}).get("n_after_year"),
                "approx_n_after_fuel": (approx_trace or {}).get("n_after_fuel"),
                "approx_n_after_cv": (approx_trace or {}).get("n_after_cv"),
                "approx_n_after_tokens": (approx_trace or {}).get("n_after_tokens"),
                "exact_year_dropped": (exact_trace or {}).get("year",{}).get("dropped"),
                "exact_fuel_relaxed": (exact_trace or {}).get("fuel",{}).get("relaxed_mode") == "ev_hybrid",
                "exact_fuel_dropped": (exact_trace or {}).get("fuel",{}).get("dropped"),
                "exact_cv_dropped": (exact_trace or {}).get("cv",{}).get("dropped"),
                "approx_year_dropped": (approx_trace or {}).get("year",{}).get("dropped"),
                "approx_fuel_relaxed": (approx_trace or {}).get("fuel",{}).get("relaxed_mode") == "ev_hybrid",
                "approx_fuel_dropped": (approx_trace or {}).get("fuel",{}).get("dropped"),
                "approx_cv_dropped": (approx_trace or {}).get("cv",{}).get("dropped"),
            }
            unmatched_rows.append(uf)
            # Guardar en audit una marca explícita de "unmatched" + diagnóstico
            ar = {"row_id_bca": r["row_id_bca"], "decision": "unmatched", "reason": "no_candidate"}
            ar.update({k: uf.get(k) for k in ["exact_fail_step","approx_fail_step","exact_year_dropped","approx_year_dropped","exact_fuel_relaxed","approx_fuel_relaxed","exact_fuel_dropped","approx_fuel_dropped","exact_cv_dropped","approx_cv_dropped"]})
            audit_rows.append(ar)
            continue

        out = {
            "row_id_bca": r["row_id_bca"],
            "marca": r["marca"],
            "modelo_base": r["modelo_base"],
            "model_bca_raw": r["model_bca_raw"],
            "year_bca": r["year_bca"],
            "cv_bca": r["cv_bca"],
            "combustible_bca": r.get("combustible_bca"),
            "row_id_ganvam": picked.get("row_id_ganvam"),
            "model_ganvam_raw": picked.get("model_ganvam_raw"),
            "version_ganvam": picked.get("version_ganvam"),
            "anio_ganvam": picked.get("anio_ganvam"),
            "cv_ganvam": picked.get("cv_ganvam"),
            "combustible_ganvam": picked.get("combustible_ganvam"),
            "precio_venta_ganvam": picked.get("precio_venta_ganvam"),
            "matching_type": picked.get("matching_type"),
            "is_approximate": picked.get("is_approximate"),
            "token_overlap": picked.get("token_overlap"),
            "token_overlap_terms": picked.get("token_overlap_terms"),
            "substring_hit": picked.get("substring_hit"),
            "stage": stage,
            "year_dropped": (exact_trace if stage=="EXACT" else approx_trace)["year"].get("dropped", False) if (exact_trace or approx_trace) else False,
            "fuel_relaxed_evhybrid": (exact_trace if stage=="EXACT" else approx_trace)["fuel"].get("relaxed_mode","") == "ev_hybrid" if (exact_trace or approx_trace) else False,
            "fuel_dropped": (exact_trace if stage=="EXACT" else approx_trace)["fuel"].get("dropped", False) if (exact_trace or approx_trace) else False,
            "cv_dropped": (exact_trace if stage=="EXACT" else approx_trace)["cv"].get("dropped", False) if (exact_trace or approx_trace) else False,
        }
        out_rows.append(out)

        # Auditoría Top‑K
        k = int(rules.get("ranking", {}).get("audit_topk", 5))
        for i, (_, row) in enumerate(cand_df.head(k).iterrows(), start=1):
            audit_rows.append({
                "row_id_bca": r["row_id_bca"],
                "cand_rank": i,
                "row_id_ganvam": row.get("row_id_ganvam"),
                "anio_ganvam": row.get("anio_ganvam"),
                "cv_ganvam": row.get("cv_ganvam"),
                "combustible_ganvam": row.get("combustible_ganvam"),
                "token_overlap": row.get("token_overlap"),
                "token_overlap_terms": row.get("token_overlap_terms"),
                "substring_hit": row.get("substring_hit"),
                "matching_type": row.get("matching_type", "CANDIDATE"),
                "stage": stage,
                "decision": "selected" if i==1 else "candidate"
            })

    matches = pd.DataFrame(out_rows).sort_values(["marca","modelo_base","row_id_bca"]).reset_index(drop=True)
    audit_topk = pd.DataFrame(audit_rows).reset_index(drop=True)
    # No escribimos out_topk desde librería.
    return matches, audit_topk

# ----------------- Utilidades de enriquecimiento -----------------

def normalize_row_id_series(s: pd.Series) -> pd.Series:
    """Normaliza un identificador potencialmente numérico (p. ej., '0.0') a string limpio ('0')."""
    s_num = pd.to_numeric(s, errors="coerce")
    # si al menos la mitad son numéricos, tratarlos como tal
    if s_num.notna().sum() >= max(1, int(0.5 * len(s))):
        out = s_num.astype("Int64").astype(str).str.replace("<NA>", "", regex=False)
        return out
    # Dejar como texto pero limpiando sufijo '.0'
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

def ensure_row_id_bca(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza 'row_id_bca' (string normalizado); si no existe, crear desde el índice."""
    out = df.copy()
    if "row_id_bca" not in out.columns:
        out["row_id_bca"] = out.index
    out["row_id_bca"] = normalize_row_id_series(out["row_id_bca"])
    return out

def compute_roi_columns(enriched: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Calcula:
      - precio_final_eur = price_col + sum(cost_cols) (solo si no existe y hay columnas)
      - margin_abs / margin_pct si existen precio_final_eur y precio_venta_ganvam
    Mantiene nombres de columnas del proceso anterior.
    """
    prm = (params or {}).get("roi", {})
    price_col = prm.get("price_col", "winning_bid")
    cost_cols = prm.get("cost_cols", []) or []
    df = enriched.copy()

    # precio_final_eur si falta
    if "precio_final_eur" not in df.columns and (price_col in df.columns or any(c in df.columns for c in cost_cols)):
        total_cost = None
        if cost_cols:
            total_cost = sum([pd.to_numeric(df.get(c), errors="coerce").fillna(0.0) for c in cost_cols])
        price_s = pd.to_numeric(df.get(price_col), errors="coerce").fillna(0.0) if price_col in df.columns else 0.0
        if total_cost is None:
            df["precio_final_eur"] = price_s
        else:
            df["precio_final_eur"] = price_s + total_cost

    # ROI
    if "precio_final_eur" in df.columns and "precio_venta_ganvam" in df.columns:
        if "margin_abs" not in df.columns:
            df["margin_abs"] = pd.to_numeric(df["precio_venta_ganvam"], errors="coerce") - pd.to_numeric(df["precio_final_eur"], errors="coerce")
        if "margin_pct" not in df.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df["margin_pct"] = df["margin_abs"] / pd.to_numeric(df["precio_final_eur"], errors="coerce")
    return df

def build_analysis_sheets(enriched: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = enriched.copy()
    sheets: Dict[str, pd.DataFrame] = {}

    # roi_por_modelo
    if {"marca","modelo_base"}.issubset(df.columns):
        grp = df.groupby(["marca","modelo_base"], dropna=False).agg(
            n=("row_id_bca","count"),
            matched=("row_id_ganvam", lambda s: int(s.notna().sum())),
            mean_margin_abs=("margin_abs","mean") if "margin_abs" in df.columns else ("row_id_bca","count"),
            mean_margin_pct=("margin_pct","mean") if "margin_pct" in df.columns else ("row_id_bca","count")
        ).reset_index()
        sheets["roi_por_modelo"] = grp

    # cobertura_por_tipo
    if "matching_type" in df.columns:
        cov = df.groupby("matching_type").size().reset_index(name="count").sort_values("count", ascending=False)
        sheets["cobertura_por_tipo"] = cov

    # flags_ratio
    flags = [c for c in ["year_dropped","fuel_relaxed_evhybrid","fuel_dropped","cv_dropped","is_approximate"] if c in df.columns]
    if flags:
        fx = df[flags].fillna(False).astype(bool)
        freq = fx.mean(numeric_only=True).to_frame("ratio").reset_index().rename(columns={"index":"flag"})
        sheets["flags_ratio"] = freq

    # top50_margin_abs / top50_margin_pct
    if {"row_id_ganvam","margin_abs","margin_pct"}.issubset(df.columns):
        top_abs = df[df["row_id_ganvam"].notna()].sort_values("margin_abs", ascending=False).head(50)
        top_pct = df[df["row_id_ganvam"].notna()].sort_values("margin_pct", ascending=False).head(50)
        sheets["top50_margin_abs"] = top_abs
        sheets["top50_margin_pct"] = top_pct

    return sheets

# ----------------- Enriquecimiento -----------------

def run_enrichment(bca_path: str, matches: pd.DataFrame, master_path: str, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Lee BCA, normaliza row_id_bca, une con `matches` por row_id_bca, calcula ROI y devuelve:
      - enriched DataFrame
      - dict de hojas de análisis (roi_por_modelo, cobertura_por_tipo, flags_ratio, top50_...).
    """
    bca = read_any(bca_path)
    bca = ensure_row_id_bca(bca)

    # Normalizar row_id_bca en matches
    mm = matches.copy()
    if "row_id_bca" not in mm.columns:
        raise RuntimeError("`matches` debe contener 'row_id_bca'")
    mm["row_id_bca"] = normalize_row_id_series(mm["row_id_bca"])

    # Merge left BCA x matches (las columnas del master ya están incluidas en matches)
    enriched = bca.merge(mm, on="row_id_bca", how="left")

    # ROI (permite derivar precio_final_eur si falta)
    enriched = compute_roi_columns(enriched, params=(params or {}))

    # Hojas de análisis
    sheets = build_analysis_sheets(enriched)

    return enriched, sheets
