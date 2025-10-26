# -*- coding: utf-8 -*-
"""
normalizacionv3_mercedes_fix.py
--------------------------------
v3 con mejoras específicas para MERCEDES:
- Regla de PRESENCIA estricta (nunca elegir un modelo no presente en el texto).
- Scoring POSICIONAL (peso decreciente para los primeros 5-6 tokens).
- Guard COUPE (GLC COUPE, etc.) y sesgo por familia (A/B/C/E/S/GLA/GLB/GLC/GLE/GLS/CLA/CLS...).
- Canonicalización display: E-VITO/ESPRINTER/ECITAN → VITO/SPRINTER/CITAN; CLASE CLA→CLA, CLASE CLS→CLS...

La firma es compatible con v3:
    normaliza_modelo_v3_mercedes_fix(marca, modelo, whitelist, display_map) -> str|None
"""

import re
import unicodedata
from typing import Dict, Set, Tuple, Optional, Iterable, List

# -------------------------
# Helpers
# -------------------------
def strip_accents_upper(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    s2 = "".join(c for c in nfkd if not unicodedata.combining(c))
    return s2.upper().strip()

def normalize_brand(marca: Optional[str]) -> str:
    if not marca or str(marca).strip() == "":
        return ""
    m = strip_accents_upper(marca)
    m = re.sub(r"\bMERCEDES[- ]?BENZ\b", "MERCEDES BENZ", m)
    m = re.sub(r"\bMERCEDES\b(?!\s*BENZ)", "MERCEDES BENZ", m)
    m = re.sub(r"\bB\.?M\.?W\b", "BMW", m)
    m = re.sub(r"\bCITRO[ËE]N\b", "CITROEN", m)
    m = re.sub(r"\bV[WÑ]\b", "VOLKSWAGEN", m)
    m = re.sub(r"\bALFA\s+ROMEO\b", "ALFA ROMEO", m)
    return m

def normalize_text(modelo: Optional[str]) -> str:
    if not modelo or str(modelo).strip() == "":
        return ""
    up = strip_accents_upper(modelo)
    up = re.sub(r"[_\.]+", " ", up)
    up = re.sub(r"\s+", " ", up).strip()
    return up

def tokenize(s: str) -> List[str]:
    if not s:
        return []
    s = re.sub(r"[^\w\s]", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

# -------------------------
# Whitelist / Display map
# -------------------------
def _detect_cols(df):
    def _up(x): 
        nfkd = unicodedata.normalize("NFKD", str(x))
        return "".join(c for c in nfkd if not unicodedata.combining(c)).upper().strip()
    cols = {c: _up(c) for c in df.columns}
    marca = next((c for c, n in cols.items() if "MARCA" == n or n.startswith("MARCA")), None)
    modelo = next((c for c, n in cols.items() if n in ("MODELO", "MODEL", "NOMBRE MODELO")), None)
    if modelo is None:
        modelo = next((c for c, n in cols.items() if "MODELO" in n or "MODEL" in n), None)
    display = next((c for c, n in cols.items() if "DISPLAY" in n or "MOSTRAR" in n or "NOMBRE COMERCIAL" in n), None)
    if marca is None or modelo is None:
        raise ValueError("No encuentro columnas de MARCA y MODELO en la whitelist")
    return marca, modelo, display

def carga_whitelist(path_xlsx: str) -> Dict[str, Set[str]]:
    import pandas as pd
    df = pd.read_excel(path_xlsx, sheet_name=0)
    c_marca, c_modelo, _ = _detect_cols(df)
    wl: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        b = normalize_brand(row[c_marca])
        m = normalize_text(row[c_modelo])
        if not b or not m:
            continue
        wl.setdefault(b, set()).add(m)
    return wl

def _v3_build_display_map(path_xlsx: str):
    import pandas as pd
    df = pd.read_excel(path_xlsx, sheet_name=0)
    c_marca, c_modelo, c_display = _detect_cols(df)
    disp = {}
    for _, row in df.iterrows():
        b_key = normalize_brand(row[c_marca])
        m_key = normalize_text(row[c_modelo])
        if not b_key or not m_key:
            continue
        if c_display and pd.notna(row[c_display]) and str(row[c_display]).strip():
            display_val = str(row[c_display]).strip()
        else:
            display_val = str(row[c_modelo]).upper().strip()
        disp[(b_key, m_key)] = display_val
    return disp

# -------------------------
# Noise/brand tokens
# -------------------------
RUIDO_GLOBAL = {
    "4X4","AWD","FWD","RWD","XDRIVE","QUATTRO","4MATIC",
    "AUTOMATIC","AUT.","AUTO","BVA","EAT","MANUAL",
    "HYBRID","PLUG-IN","PHEV","ELECTRIC","DIESEL","GASOLINA","PETROL",
    "SPORT","M-SPORT","AMG","GTI","ST","RS",
    "GRAND","GRAN","BUSINESS","LIFE","LOUNGE","CHIC",
    "TOURER","ACTIVE","XLINE","URBAN",
    "CROSSBACK","CROSS","STEPWAY","COUPE"
}
RUIDO_EXCEPTIONS = {
    "MERCEDES BENZ": {"COUPE"},
    "MINI": {"ELECTRIC"}
}

def _filter_noise(tokens: Iterable[str], marca_norm: str) -> List[str]:
    ex = RUIDO_EXCEPTIONS.get(marca_norm, set())
    out = []
    mercedes_noise = {"CLASE", "CLASSE", "CLASS"} if marca_norm == "MERCEDES BENZ" else set()
    for t in tokens:
        t_up = strip_accents_upper(t)
        if t_up in ex:
            out.append(t_up)
        elif t_up in RUIDO_GLOBAL or t_up in mercedes_noise:
            continue
        else:
            out.append(t_up)
    return out

BRAND_TOKENS = {
    "ABARTH","ALFA","ROMEO","ALFA ROMEO","ASTON","MARTIN","ASTON MARTIN","AUDI","BMW","BYD","CITROEN","CUPRA",
    "DACIA","DS","FERRARI","FIAT","FORD","HONDA","HYUNDAI","INFINITI","JAGUAR","JEEP","KIA","LADA","LANCIA","LAND","ROVER",
    "LAND ROVER","LEXUS","MERCEDES","BENZ","MERCEDES BENZ","MG","MINI","MITSUBISHI","NISSAN","OPEL","PEUGEOT","PORSCHE",
    "RENAULT","ROLLS","ROYCE","ROLLS ROYCE","SEAT","SKODA","SMART","SSANGYONG","SUBARU","SUZUKI","TESLA","TOYOTA","VOLKSWAGEN","VW","VOLVO"
}

def _first_non_brand_token(tokens: List[str], marca_norm: str):
    if not tokens:
        return None
    brand_parts = set(str(marca_norm).upper().split())
    if marca_norm == "MERCEDES BENZ":
        brand_parts.update({"MERCEDES", "BENZ"})
    if marca_norm == "ALFA ROMEO":
        brand_parts.update({"ALFA", "ROMEO"})
    for t in tokens:
        tu = t.upper()
        if tu not in brand_parts and tu not in BRAND_TOKENS:
            return t
    return None

# -------------------------
# Scoring
# -------------------------
_POS_WEIGHTS = [1.0, 0.85, 0.7, 0.55, 0.4, 0.3]

def _alpha_token(s: str) -> bool:
    return s.isalpha()

def _score_candidate_general(model_in_norm: str, cand_norm: str, marca_norm: str = "") -> float:
    tokens_in = list(tokenize(model_in_norm))
    tokens_cand = list(tokenize(cand_norm))
    if not tokens_cand:
        return -1.0

    hits = 0
    for tc in tokens_cand:
        if any(ti.startswith(tc) or tc.startswith(ti) for ti in tokens_in):
            hits += 1
    recall = hits / max(len(tokens_cand), 1)

    ti0 = _first_non_brand_token(tokens_in, marca_norm)
    tc0 = _first_non_brand_token(tokens_cand, marca_norm)
    pref = 1.0 if ti0 and tc0 and (ti0.startswith(tc0) or tc0.startswith(ti0)) else 0.0

    length_penalty = 1.0 / (1.0 + 0.15 * max(len(cand_norm), 1))

    score = 0.55 * recall + 0.35 * pref + 0.10 * length_penalty
    return score

def _score_candidate_mercedes(model_in_norm: str, cand_norm: str) -> float:
    tokens_in = tokenize(model_in_norm)
    tokens_cand = tokenize(cand_norm)
    tokens_cand_core = [t for t in tokens_cand if t not in {"MERCEDES","BENZ","MERCEDES BENZ","CLASE","CLASSE","CLASS"}]
    if not tokens_cand_core:
        tokens_cand_core = tokens_cand
    pos_score = 0.0
    first_token_bonus = 0.0
    for tc in tokens_cand_core:
        found_pos = None
        for i, ti in enumerate(tokens_in):
            if _alpha_token(tc):
                cond = (ti == tc)  # exact match for pure alpha tokens
            else:
                cond = (ti == tc or ti.startswith(tc) or tc.startswith(ti))
            if cond:
                found_pos = i
                break
        if found_pos is None:
            return -1.0
        weight = _POS_WEIGHTS[found_pos] if found_pos < len(_POS_WEIGHTS) else 0.2
        pos_score += weight
        if found_pos == 0:
            first_token_bonus = 0.25
    pos_score = pos_score / max(1, len(tokens_cand_core))
    length_penalty = 1.0 / (1.0 + 0.2 * len(" ".join(tokens_cand_core)))
    return 0.75 * pos_score + 0.20 * length_penalty + 0.05 * first_token_bonus

# -------------------------
# Canonical display
# -------------------------
CANONICAL_DISPLAY = {
    "DS": {
        "DS3": "DS 3", "DS4": "DS 4", "DS5": "DS 5", "DS7": "DS 7",
        "DS-3": "DS 3", "DS-4": "DS 4", "DS-5": "DS 5", "DS-7": "DS 7",
    },
    "MITSUBISHI": {"L-200": "L200", "L200": "L200"},
    "NISSAN": {"370-Z": "370Z", "370Z": "370Z", "GT-R": "GTR", "GTR": "GTR"},
    "TOYOTA": {"GT-86": "GT86", "GT86": "GT86"},
    "MERCEDES BENZ": {
        "EVITO": "VITO", "E-VITO": "VITO", "E VITO": "VITO",
        "ESPRINTER": "SPRINTER", "E-SPRINTER": "SPRINTER", "E SPRINTER": "SPRINTER",
        "ECITAN": "CITAN", "E-CITAN": "CITAN", "E CITAN": "CITAN",
        "CLASE CLA": "CLA",
        "CLASE CLS": "CLS",
        "CLASE SLK": "SLK",
        "CLASE SL": "SL",
        "CLASE SLC": "SLC",
        "CLASE CLK": "CLK",
        "CLASE GLK": "GLK",
        "CLASE GL": "GL",
        "CLASE CLA COUPE": "CLA COUPE",
        "CLASE CLS COUPE": "CLS COUPE",
    },
}

def _unify_display(marca_norm: str, display: str) -> str:
    if not display:
        return display
    disp = display.upper().strip()
    return CANONICAL_DISPLAY.get(marca_norm, {}).get(disp, disp)

# -------------------------
# Brand guards
# -------------------------
def _brand_guard(marca_norm: str, modelo_in_norm: str, cands_all: Set[str]) -> Optional[str]:
    up = modelo_in_norm

    # MERCEDES BENZ
    if marca_norm == "MERCEDES BENZ":
        # If COUPE appears, restrict to candidates with COUPE
        if "COUPE" in up:
            coupes = [c for c in cands_all if "COUPE" in c]
            if coupes:
                best_c, best_s = None, -1.0
                for c in coupes:
                    s = _score_candidate_mercedes(up, c)
                    if s > best_s:
                        best_s, best_c = s, c
                return best_c
        # Explicit safeguard
        if re.search(r"\bGLC\b.*\bCOUPE\b", up) and "GLC COUPE" in cands_all:
            return "GLC COUPE"

        # Family soft-bias based on first non-brand token
        toks = list(_filter_noise(tokenize(up), marca_norm))
        ti0 = _first_non_brand_token(toks, marca_norm)
        if ti0:
            fam = ti0.upper()
            fam_map = {
                "A": "CLASE A",
                "B": "CLASE B",
                "C": "CLASE C",
                "E": "CLASE E",
                "S": "CLASE S",
                "CLA": "CLA",
                "CLS": "CLASE CLS",
                "CLK": "CLASE CLK",
                "SLK": "CLASE SLK",
                "SL": "CLASE SL",
                "SLC": "CLASE SLC",
                "GL": "CLASE GL",
                "GLA": "GLA",
                "GLB": "GLB",
                "GLC": "GLC",
                "GLE": "GLE",
                "GLS": "GLS",
                "GLK": "CLASE GLK",
            }
            if fam in fam_map:
                family_token = fam_map[fam]
                fam_cands = [c for c in cands_all if family_token in c]
                if fam_cands:
                    best_c, best_s = None, -1.0
                    for c in fam_cands:
                        s = _score_candidate_mercedes(up, c)
                        if s > best_s:
                            best_s, best_c = s, c
                    return best_c

    # DS: DS3/4/5/7 variants
    if marca_norm == "DS":
        up_no_space = re.sub(r"\s+", "", modelo_in_norm)
        m = re.search(r"\bDS[- ]?([357])\b|\bDS([357])\b", modelo_in_norm) or re.search(r"^DS([357])\b", up_no_space)
        if m:
            d = (m.group(1) or m.group(2)).strip()
            if f"DS {d}" in cands_all:
                return f"DS {d}"
            if f"DS{d}" in cands_all:
                return f"DS{d}"

    # BMW
    if marca_norm == "BMW":
        if "ACTIVE TOURER" in up or re.search(r"\b225X?E\b", up):
            if "SERIE 2" in cands_all:
                return "SERIE 2"
        if re.search(r"\bI3S\b", up) and "I3" in cands_all:
            return "I3"
        if re.search(r"\b1(16|18|20)[ID]\b|\b118I[A-Z]*\b", up) and "SERIE 1" in cands_all:
            return "SERIE 1"

    # NISSAN
    if marca_norm == "NISSAN":
        if re.search(r"\bX[- ]?TRAIL\b|\bXTRAIL\b", up) and "X-TRAIL" in cands_all:
            return "X-TRAIL"

    # KIA
    if marca_norm == "KIA":
        if re.search(r"\bPRO[\s_]?CEE[’'`´]?D\b", up) and "PROCEED" in cands_all:
            return "PROCEED"

    # MINI: Electric / Countryman / Clubman / Paceman
    if marca_norm == "MINI":
        # 'up' y 'cands_all' ya están definidos en el contexto del guard
        # Electric (evidencia explícita)
        if re.search(r"\b(ELECTRIC|SE|COOPER\s*SE|E[- ]?MINI)\b", up) and "MINI ELECTRIC" in cands_all:
            return "MINI ELECTRIC"
        # Countryman (siempre priorizar cuando aparece)
        if "COUNTRYMAN" in up and "COUNTRYMAN" in cands_all:
            return "COUNTRYMAN"
        # Clubman (siempre priorizar cuando aparece)
        if "CLUBMAN" in up and "CLUBMAN" in cands_all:
            return "CLUBMAN"
        # Paceman (siempre priorizar cuando aparece)
        if "PACEMAN" in up and "PACEMAN" in cands_all:
            return "PACEMAN"
        # ONE sólo si no hay señales de Electric/SE (regla ya existente)
        if "ONE" in up and "ONE" in cands_all:
            if not re.search(r"\b(ELECTRIC|SE|COOPER\s*SE|E[- ]?MINI)\b", up):
                return "ONE"

    # SMART
    if marca_norm == "SMART":
        if "FORTWO" in up and "FORTWO" in cands_all:
            return "FORTWO"
        if "FORFOUR" in up and "FORFOUR" in cands_all:
            return "FORFOUR"

    # HONDA
    if marca_norm == "HONDA":
        if re.search(r"\bCR[- ]?Z\b", up) and "CR-Z" in cands_all:
            return "CR-Z"
        if re.search(r"\bHR[- ]?V\b", up) and "HR-V" in cands_all:
            return "HR-V"

    # HYUNDAI
    if marca_norm == "HYUNDAI":
        if re.search(r"\bH[- ]?1\b", up) and "H-1" in cands_all:
            return "H-1"

    # FORD
    if marca_norm == "FORD":
        if re.search(r"\bB[- ]?MAX\b", up) and "B-MAX" in cands_all:
            return "B-MAX"

    # CITROEN
    if marca_norm == "CITROEN":
        if re.search(r"\bC[- ]?E?LYS[ÉE]E\b|\bCELYS[ÉE]E\b", up) and "C-ELYSEE" in cands_all:
            return "C-ELYSEE"

    # ALFA ROMEO
    if marca_norm == "ALFA ROMEO":
        if "TONALE" in up and "TONALE" in cands_all:
            return "TONALE"

    return None

# -------------------------
# Main normalizer
# -------------------------
def normaliza_modelo_v3_mercedes_fix(marca: Optional[str],
                                     modelo: Optional[str],
                                     whitelist: Dict[str, Set[str]],
                                     display_map: Dict[Tuple[str, str], str]) -> Optional[str]:
    if marca is None and modelo is None:
        return None

    marca_n = normalize_brand(marca)
    modelo_in = normalize_text(modelo)
    if not marca_n or not modelo_in:
        return None

    # Mercedes vans: quitar prefijo E-
    if marca_n == "MERCEDES BENZ":
        modelo_in = re.sub(r"\bE[- ]?(VITO|SPRINTER|CITAN)\b", r"\1", modelo_in)

    # MAZDA passthrough
    if marca_n == "MAZDA":
        m = re.search(r'\bMAZDA[\s\-_]*([36])(?!\d)\b', modelo_in)
        if m:
            target = m.group(1)
            if target in whitelist.get("MAZDA", set()):
                return display_map.get(("MAZDA", target), target)

    toks_in = list(_filter_noise(tokenize(modelo_in), marca_n))
    modelo_in_nr = " ".join(toks_in) if toks_in else modelo_in

    cands_all = set(whitelist.get(marca_n, set()))
    if not cands_all:
        return None

    # Guards first
    guard = _brand_guard(marca_n, modelo_in_nr, cands_all)
    if guard:
        display_out = display_map.get((marca_n, guard), guard)
        return _unify_display(marca_n, display_out)

    # Mercedes: strict presence + positional scoring
    if marca_n == "MERCEDES BENZ":
        def _presence_ok(model_in_norm: str, cand_norm: str) -> bool:
            tokens_in = set(tokenize(model_in_norm))
            tokens_cand = [t for t in tokenize(cand_norm) if t not in {"MERCEDES","BENZ","MERCEDES BENZ","CLASE","CLASSE","CLASS"}]
            if not tokens_cand:
                return False
            for tc in tokens_cand:
                if _alpha_token(tc):
                    ok = tc in tokens_in
                else:
                    ok = any(ti == tc or ti.startswith(tc) or tc.startswith(ti) for ti in tokens_in)
                if not ok:
                    return False
            return True

        cands_present = [c for c in cands_all if _presence_ok(modelo_in_nr, c)]
        if "COUPE" in modelo_in_nr:
            cands_present = [c for c in cands_present if "COUPE" in c] or [c for c in cands_present]
        if cands_present:
            best_norm = None
            best_score = -1.0
            for cand_norm in cands_present:
                s = _score_candidate_mercedes(modelo_in_nr, cand_norm)
                if s > best_score:
                    best_score = s
                    best_norm = cand_norm
            if best_norm:
                display_out = display_map.get((marca_n, best_norm), best_norm)
                return _unify_display(marca_n, display_out)

        # fallback: general scoring but still require presence
        best_norm = None
        best_score = -1.0
        for cand_norm in cands_all:
            if not _presence_ok(modelo_in_nr, cand_norm):
                continue
            s = _score_candidate_general(modelo_in_nr, cand_norm, marca_n)
            if s > best_score:
                best_score = s
                best_norm = cand_norm
        if best_norm:
            display_out = display_map.get((marca_n, best_norm), best_norm)
            return _unify_display(marca_n, display_out)
        return None

    # Other brands: general scoring
    best_norm = None
    best_score = -1.0
    for cand_norm in cands_all:
        score = _score_candidate_general(modelo_in_nr, cand_norm, marca_n)
        if score > best_score:
            best_score = score
            best_norm = cand_norm

    if best_norm is None:
        return None
    display_out = display_map.get((marca_n, best_norm), best_norm)
    return _unify_display(marca_n, display_out)
