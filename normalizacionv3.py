
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalizacionv3_patched_full.py
--------------------------------
Helpers v3 + brand-guards quirúrgicos consolidados:

- Normalización básica (marca/modelo, tokenización, ruido).
- CANONICAL_DISPLAY enriquecido (incluye VW ID.7 y variantes).
- _brand_guard con reglas tempranas:
    * BMW: iXn / Xn / Zn cuando aparecen explícitos.
    * MERCEDES: GLA/GLB/GLC/GLE/GLS blindados; GLC COUPE si aparece.
    * VOLVO: V/S/XC + 2 dígitos → base exacta (p.ej. 'V 60 Cross Country' -> V60).
    * MAZDA: 'MAZDA2' → '2' y 'MAZDA5' → '5' (sin colisión con CX-5/MX-5).
    * DS/DR/otros: alias habituales.

No expone un "normalizador v3" completo, porque el orquestador fusión lo usa como helpers.
"""

import re, unicodedata
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
    m = re.sub(r"\bDS\s+AUTOMOBILES\b", "DS", m)
    m = re.sub(r"\bDR\s+AUTOMOBILES\b", "DR", m)
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
# Canonical display
# -------------------------
CANONICAL_DISPLAY = {
    "AUDI": {"E-TRON": "E TRON", "E TRON": "E TRON"},
    "DS": {
        "DS3": "DS 3", "DS4": "DS 4", "DS5": "DS 5", "DS7": "DS 7",
        "DS-3": "DS 3", "DS-4": "DS 4", "DS-5": "DS 5", "DS-7": "DS 7",
    },
    "MITSUBISHI": {"L-200": "L200", "L200": "L200"},
    "NISSAN": {"370-Z": "370Z", "370Z": "370Z", "GT-R": "GTR", "GTR": "GTR"},
    "TOYOTA": {"GT-86": "GT86", "GT86": "GT86"},
    "FIAT": {"500 E":"500E","500E":"500E"},
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
    "VOLKSWAGEN": {
        "T ROC": "T-ROC", "TROC": "T-ROC",
        "T CROSS": "T-CROSS", "TCROSS": "T-CROSS", "T-CROS": "T-CROSS",
        "ID BUZZ": "ID. BUZZ", "ID. BUZZ": "ID. BUZZ",
        "ID 3":"ID.3","ID3":"ID.3","ID-3":"ID.3","ID . 3":"ID.3","id3":"ID.3",
        "ID 4":"ID.4","ID4":"ID.4","ID-4":"ID.4","ID . 4":"ID.4","id4":"ID.4",
        "ID 5":"ID.5","ID5":"ID.5","ID-5":"ID.5","ID . 5":"ID.5","id5":"ID.5",
        "ID 7":"ID.7","ID7":"ID.7","ID-7":"ID.7","ID . 7":"ID.7","id7":"ID.7"
    },
    "DR": {
        "DR3": "DR 3", "DR4": "DR 4", "DR5": "DR 5",
    }
}

def _unify_display(marca_norm: str, display: str) -> str:
    if not display:
        return display
    disp = display.upper().strip()
    return CANONICAL_DISPLAY.get(marca_norm, {}).get(disp, disp)

# -------------------------
# Brand guards (quirúrgicos)
# -------------------------
def _brand_guard(marca_norm: str, modelo_in_norm: str, cands_all: Set[str]) -> Optional[str]:
    up = modelo_in_norm

    # -------- BMW: iXn / Xn / Zn --------
    if marca_norm == "BMW":
        m = re.search(r'\bIX\s*([1-7])\b', up)
        if m:
            ix = f"IX{m.group(1)}"
            if ix in cands_all:
                return ix
        if re.search(r'\bIX\b', up) and "IX" in cands_all:
            return "IX"
        m = re.search(r'\bX\s*([1-7])\b', up)
        if m:
            xr = f"X{m.group(1)}"
            if xr in cands_all:
                return xr
        m = re.search(r'\bZ\s*([1-8])\b', up)
        if m:
            zr = f"Z{m.group(1)}"
            if zr in cands_all:
                return zr

    # -------- MERCEDES BENZ --------
    if marca_norm == "MERCEDES BENZ":
        # GLC COUPE explícito
        if re.search(r"\bGLC\b.*\bCOUPE\b", up) and "GLC COUPE" in cands_all:
            return "GLC COUPE"
        # GLA/GLB/GLC/GLE/GLS blindados
        for suv in ("GLA","GLB","GLC","GLE","GLS"):
            if re.search(rf'\b{suv}\b', up) and suv in cands_all:
                return suv

    # -------- VOLVO --------
    if marca_norm == "VOLVO":
        m = re.search(r'\b(V|S|XC)\s*-?\s*(\d{2})\b', up)
        if m:
            fam, num = m.group(1), m.group(2)
            target = f"{fam}{num}"
            if target in cands_all:
                return target

    # -------- MAZDA (2/5 sin colisión con CX-5/MX-5) --------
    if marca_norm == "MAZDA":
        if re.search(r'\bMAZDA\s*2\b|\bMAZDA2\b', up) and not re.search(r'\b(CX|MX)\s*-?\s*2\b', up):
            if "2" in cands_all:
                return "2"
        if re.search(r'\bMAZDA\s*5\b|\bMAZDA5\b', up) and not re.search(r'\b(CX|MX)\s*-?\s*5\b', up):
            if "5" in cands_all:
                return "5"
        if re.search(r'\bMAZDA\s*3\b|\bMAZDA3\b', up) and not re.search(r'\b(CX|MX)\s*-?\s*3\b', up):
            if "3" in cands_all:
                return "3"
        if re.search(r'\bMAZDA\s*6\b|\bMAZDA6\b', up) and not re.search(r'\b(CX|MX)\s*-?\s*6\b', up):
            if "6" in cands_all:
                return "6"




    # -------- DS --------
    if marca_norm == "DS":
        up_no_space = re.sub(r"\s+", "", up)
        m = re.search(r"\bDS[- ]?([3457])\b|\bDS([3457])\b", up) or re.search(r"^DS([3457])\b", up_no_space)
        if m:
            d = (m.group(1) or m.group(2)).strip()
            if f"DS {d}" in cands_all:
                return f"DS {d}"
            if f"DS{d}" in cands_all:
                return f"DS{d}"

    # -------- DR --------
    if marca_norm == "DR":
        m = re.search(r'\bDR[\s\-]?(\d)(?:[.,](\d))?\b', up)
        if m:
            cand = f"DR {m.group(1)}" + (f".{m.group(2)}" if m.group(2) else "")
            if cand in cands_all:
                return cand

    # -------- Otros alias puntuales --------
    if marca_norm == "NISSAN":
        if re.search(r"\bX[- ]?TRAIL\b|\bXTRAIL\b", up) and "X-TRAIL" in cands_all:
            return "X-TRAIL"
    if marca_norm == "KIA":
        if re.search(r"\bPRO[\s_]?CEE[’'`´]?D\b", up) and "PROCEED" in cands_all:
            return "PROCEED"
    if marca_norm == "SMART":
        if "FORTWO" in up and "FORTWO" in cands_all: return "FORTWO"
        if "FORFOUR" in up and "FORFOUR" in cands_all: return "FORFOUR"
    if marca_norm == "HONDA":
        if re.search(r"\bCR[- ]?Z\b", up) and "CR-Z" in cands_all: return "CR-Z"
        if re.search(r"\bHR[- ]?V\b", up) and "HR-V" in cands_all: return "HR-V"
    if marca_norm == "HYUNDAI":
        if re.search(r"\bH[- ]?1\b", up) and "H-1" in cands_all: return "H-1"
    if marca_norm == "FORD":
        if re.search(r"\bB[- ]?MAX\b", up) and "B-MAX" in cands_all: return "B-MAX"
    if marca_norm == "CITROEN":
        if re.search(r"\bC[- ]?E?LYS[ÉE]E\b|\bCELYS[ÉE]E\b", up) and "C-ELYSEE" in cands_all: return "C-ELYSEE"
    if marca_norm == "ALFA ROMEO":
        if "TONALE" in up and "TONALE" in cands_all: return "TONALE"

    return None
