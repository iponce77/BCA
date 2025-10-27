
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalizacion_fusion_plug_prod_full_patched.py  (regex fixed)
"""
from __future__ import annotations
import json, re, sys, argparse
from typing import Dict, Set, Tuple, Optional, Iterable, List
from pathlib import Path

# Dependencias locales
sys.path.append("/mnt/data")
import pandas as pd
import normalizacionv2_legacy as v2
import normalizacionv3 as v3

def _norm(s: Optional[str]) -> str:
    if s is None: return ""
    return v3.normalize_text(s)

def _tokens(s: str) -> List[str]:
    return list(v3.tokenize(s))

def _filter_noise(tokens: Iterable[str], brand_norm: str) -> List[str]:
    return list(v3._filter_noise(tokens, brand_norm))

def _first_non_brand(tokens: Iterable[str], brand_norm: str) -> Optional[str]:
    return v3._first_non_brand_token(tokens, brand_norm)

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter/union if union else 0.0

def _presence_ok(brand: str, model_in_nr: str, cand: str) -> bool:
    tc = [t for t in _tokens(cand) if t not in {"MERCEDES","BENZ","MERCEDES BENZ","CLASE","CLASSE","CLASS"}]
    if not tc: return False
    ti = _tokens(model_in_nr)
    if not ti: return False
    if brand == "MERCEDES BENZ":
        for t in tc:
            if t.isalpha():
                if t not in ti:
                    return False
            else:
                if not any((x == t) or x.startswith(t) or t.startswith(x) for x in ti):
                    return False
        return True
    return set(tc).issubset(set(ti))

def _pref_pos_score(model_in_nr: str, cand: str) -> float:
    POS_W = [1.0, 0.85, 0.7, 0.55, 0.4, 0.3]
    ti = _tokens(model_in_nr)
    tc = [t for t in _tokens(cand) if t not in {"MERCEDES","BENZ","MERCEDES BENZ","CLASE","CLASSE","CLASS"}]
    if not ti or not tc: return 0.0
    acc = 0.0; n=0
    for t in tc:
        pos = None
        for i, x in enumerate(ti):
            if x == t or x.startswith(t) or t.startswith(x):
                pos = i; break
        if pos is None:
            continue
        acc += POS_W[pos] if pos < len(POS_W) else 0.2
        n += 1
    return (acc / n) if n>0 else 0.0

def _detect_family(brand: str, model_in_nr: str) -> Optional[str]:
    up = model_in_nr
    if brand == "BMW":
        m = re.search(r"\b([1-8])\d{2}[A-Z]?\b", up)
        if m: return f"SERIE {m.group(1)}"
    if brand == "MERCEDES BENZ":
        toks = _tokens(up)
        fams = {"A":"CLASE A","B":"CLASE B","C":"CLASE C","E":"CLASE E","S":"CLASE S",
                "CLA":"CLA","CLS":"CLASE CLS","GLA":"GLA","GLB":"GLB","GLC":"GLC","GLE":"GLE","GLS":"GLS"}
        t0 = _first_non_brand(toks, brand)
        if t0:
            t0u = t0.upper()
            if t0u in fams: return fams[t0u]
    if brand == "VOLKSWAGEN":
        m = re.search(r"\bID\s*[\.\-]?([3457])\b", up)
        if m: return f"ID.{m.group(1)}"
    if brand == "MAZDA":
        m = re.search(r"\b(CX)\s*-?\s*(\d{1,2})\b", up)
        if m: return f"{m.group(1)}-{m.group(2)}"
        m = re.search(r"\b(MX)\s*-?\s*(\d{1,2})\b", up)
        if m: return f"{m.group(1)}-{m.group(2)}"
        m = re.search(r"\bMAZDA\s*([2356])\b|\bMAZDA([2356])\b", up)
        if m:
            d = (m.group(1) or m.group(2))
            return d
    if brand == "DS":
        m = re.search(r"\bDS[\s\-]?([3457])\b", up)
        if m: return f"DS {m.group(1)}"
    if brand == "DR":
        m = re.search(r"\bDR[\s\-]?(\d)(?:[.,](\d))?\b", up)
        if m:
            return f"DR {m.group(1)}" + (f".{m.group(2)}" if m.group(2) else "")
    return None

def _canonical_alias(brand: str, disp: str) -> str:
    disp_up = disp.upper().strip()
    extra = {
        "VOLKSWAGEN": {
            "ID 3":"ID.3","ID3":"ID.3","ID-3":"ID.3","ID . 3":"ID.3",
            "ID 4":"ID.4","ID4":"ID.4","ID-4":"ID.4","ID . 4":"ID.4",
            "ID 5":"ID.5","ID5":"ID.5","ID-5":"ID.5","ID . 5":"ID.5",
            "ID 7":"ID.7","ID7":"ID.7","ID-7":"ID.7","ID . 7":"ID.7"
        },
        "MAZDA": {"MX 5":"MX-5","CX 5":"CX-5","CX 30":"CX-30","MX 30":"MX-30","CX 60":"CX-60"},
        "MG": {"EHS":"HS"},
        "DR": {"DR3":"DR 3","DR5":"DR 5","DR4":"DR 4"},
        "DS": {"DS3":"DS 3","DS4":"DS 4","DS5":"DS 5","DS7":"DS 7"},
    }
    base = getattr(v3, "CANONICAL_DISPLAY", {}).get(brand, {})
    if disp_up in base: return base[disp_up]
    if brand in extra and disp_up in extra[brand]: return extra[brand][disp_up]
    return disp

def _candidate_pool(brand_raw: str, model_raw: str,
                    wl3: Dict[str, Set[str]], disp3: Dict[Tuple[str,str], str],
                    wl2: Dict[str, Set[str]]) -> Tuple[str, str, List[str], Optional[str], Optional[str]]:
    brand = v3.normalize_brand(brand_raw)
    model_in = v3.normalize_text(model_raw)

    if brand == "MERCEDES BENZ":
        model_in = re.sub(r"\bE[- ]?(VITO|SPRINTER|CITAN)\b", r"\1", model_in)

    toks = _filter_noise(_tokens(model_in), brand)
    model_in_nr = " ".join(toks) if toks else model_in

    cands = set(wl3.get(brand, set()))
    v2_cand = None
    try:
        v2_cand = v2.normaliza_modelo(brand_raw, model_raw, wl2)
    except Exception:
        v2_cand = None
    if v2_cand: cands.add(_norm(v2_cand))

    fam = _detect_family(brand, model_in_nr)
    if fam and fam in cands: cands.add(fam)

    if brand == "MERCEDES BENZ":
        cands = {c for c in cands if _presence_ok(brand, model_in_nr, c)}
        if "COUPE" in model_in_nr:
            cands_coupe = {c for c in cands if "COUPE" in c}
            if cands_coupe:
                cands = cands_coupe

    if brand == "MINI":
        up = model_in_nr
        if re.search(r'\bPACEMAN\b', up):
            pm = {c for c in cands if "PACEMAN" in c}
            if pm: cands = pm
        elif re.search(r'\bCOUNTRYMAN\b', up) or re.search(r'\bCOUNTYMAN\b', up):
            cm = {c for c in cands if "COUNTRYMAN" in c}
            if cm: cands = cm
        elif re.search(r'\bCLUBMAN\b', up):
            cl = {c for c in cands if "CLUBMAN" in c}
            if cl: cands = cl
        elif re.search(r'\b(ELECTRIC|SE|COOPER\s*SE|E[-\s]?MINI)\b', up):
            if "MINI ELECTRIC" in cands:
                cands = {"MINI ELECTRIC"}

    return brand, model_in_nr, sorted(cands), v2_cand, fam

def _score_features(brand: str, model_in_nr: str, cand: str,
                    v2_cand: Optional[str], fam_hint: Optional[str], pool_for_guard: Set[str]) -> Dict[str, float]:
    tokens_in = _tokens(model_in_nr)
    tokens_c = _tokens(cand)
    hits = 0
    for tc in tokens_c:
        if any(ti.startswith(tc) or tc.startswith(ti) for ti in tokens_in):
            hits += 1
    recall = hits / max(len(tokens_c), 1)

    ti0 = _first_non_brand(tokens_in, brand)
    tc0 = _first_non_brand(tokens_c, brand)
    pref = 1.0 if ti0 and tc0 and (ti0.startswith(tc0) or tc0.startswith(ti0)) else 0.0

    length_penalty = 1.0 / (1.0 + 0.15 * max(len(cand), 1))
    subset_bonus = 1.0 if set(tokens_c) and set(tokens_c).issubset(set(tokens_in)) else 0.0
    digits_bonus = 1.0 if (any(ch.isdigit() for ch in "".join(tokens_c)) and any(ch.isdigit() for ch in "".join(tokens_in))) else 0.0

    guard_choice = None
    try:
        guard_choice = v3._brand_guard(brand, model_in_nr, set(pool_for_guard))
    except Exception:
        guard_choice = None
    guard_bonus = 0.0
    if guard_choice:
        if _norm(guard_choice) == _norm(cand):
            guard_bonus = 1.0
        else:
            guard_bonus = -0.2

    v2_flag = 1.0 if (v2_cand and _norm(v2_cand) == _norm(cand)) else 0.0
    fam_flag = 1.0 if (fam_hint and fam_hint in cand) else 0.0
    lex = _jaccard(tokens_in, tokens_c)

    present = 1.0 if _presence_ok(brand, model_in_nr, cand) else 0.0
    first_token = 1.0 if (ti0 and tc0 and ti0 == tc0) else 0.0
    pref_pos = _pref_pos_score(model_in_nr, cand)

    coupe = 0.0
    if "COUPE" in model_in_nr:
        if "COUPE" in cand:
            coupe = 1.0
        else:
            coupe = -1.0

    return {
        "recall": recall, "pref": pref, "length": length_penalty,
        "subset": subset_bonus, "digits": digits_bonus,
        "guard": guard_bonus, "v2flag": v2_flag, "family": fam_flag, "lex": lex,
        "present": present, "first_token": first_token, "pref_pos": pref_pos, "coupe": coupe
    }

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

def _brand_weights(weights: Dict[str, Dict[str,float]], brand: str) -> Dict[str,float]:
    w = weights.get(brand, None)
    if not w: return weights["default"]
    out = weights["default"].copy()
    out.update(w)
    return out

def _linear_score(feats: Dict[str,float], w: Dict[str,float]) -> float:
    return sum(feats[k]*w.get(k,0.0) for k in feats.keys())

def _hard_patches(brand: str, model_in_nr: str, wl3: Dict[str, Set[str]], disp3: Dict[Tuple[str,str], str]):
    up = model_in_nr

    # MERCEDES: MARCO POLO
    if brand == "MERCEDES BENZ" and re.search(r'\bMARCO\s+POLO\b', up):
        if "MARCO POLO" in wl3.get(brand, set()):
            disp = disp3.get((brand, "MARCO POLO"), "MARCO POLO")
            return _canonical_alias(brand, disp)

    # LAND ROVER: D350 -> DEFENDER
    if brand == "LAND ROVER" and re.search(r'\bD350\b', up):
        if "DEFENDER" in wl3.get(brand, set()):
            disp = disp3.get((brand, "DEFENDER"), "DEFENDER")
            return _canonical_alias(brand, disp)

    # MG: EHS -> HS
    if brand == "MG" and re.search(r'\bEHS\b', up):
        if "HS" in wl3.get(brand, set()):
            disp = disp3.get((brand, "HS"), "HS")
            return _canonical_alias(brand, disp)

    # BMW: iXn / Xn / Zn (retorno temprano)
    if brand == "BMW":
        m = re.search(r'\bIX\s*([1-7])\b', up)
        if m and f"IX{m.group(1)}" in wl3.get(brand, set()):
            cand = f"IX{m.group(1)}"; disp = disp3.get((brand, cand), cand); return _canonical_alias(brand, disp)
        if re.search(r'\bIX\b', up) and "IX" in wl3.get(brand, set()):
            disp = disp3.get((brand, "IX"), "IX"); return _canonical_alias(brand, disp)
        m = re.search(r'\bX\s*([1-7])\b', up)
        if m and f"X{m.group(1)}" in wl3.get(brand, set()):
            cand = f"X{m.group(1)}"; disp = disp3.get((brand, cand), cand); return _canonical_alias(brand, disp)
        m = re.search(r'\bZ\s*([1-8])\b', up)
        if m and f"Z{m.group(1)}" in wl3.get(brand, set()):
            cand = f"Z{m.group(1)}"; disp = disp3.get((brand, cand), cand); return _canonical_alias(brand, disp)

    # VOLVO: V/S/XC + 2 dígitos → base exacta
    if brand == "VOLVO":
        m = re.search(r'\b(V|S|XC)\s*-?\s*(\d{2})\b', up)
        if m:
            cand = f"{m.group(1)}{m.group(2)}"
            if cand in wl3.get(brand, set()):
                disp = disp3.get((brand, cand), cand)
                return _canonical_alias(brand, disp)

    # MAZDA: Mazda2/Mazda5 explícitos (sin colisionar con CX-5/MX-5)
    if brand == "MAZDA":
        if re.search(r'\bMAZDA\s*2\b|\bMAZDA2\b', up) and not re.search(r'\b(CX|MX)\s*-?\s*2\b', up):
            if "2" in wl3.get(brand, set()):
                disp = disp3.get((brand, "2"), "2"); return _canonical_alias(brand, disp)
        if re.search(r'\bMAZDA\s*5\b|\bMAZDA5\b', up) and not re.search(r'\b(CX|MX)\s*-?\s*5\b', up):
            if "5" in wl3.get(brand, set()):
                disp = disp3.get((brand, "5"), "5"); return _canonical_alias(brand, disp)
        if re.search(r'\bMAZDA\s*3\b|\bMAZDA3\b', up) and not re.search(r'\b(CX|MX)\s*-?\s*3\b', up):
            if "3" in wl3.get(brand, set()):
                disp = disp3.get((brand, "3"), "3"); return _canonical_alias(brand, disp)
        if re.search(r'\bMAZDA\s*6\b|\bMAZDA6\b', up) and not re.search(r'\b(CX|MX)\s*-?\s*6\b', up):
            if "6" in wl3.get(brand, set()):
                disp = disp3.get((brand, "6"), "6"); return _canonical_alias(brand, disp)


    # VOLKSWAGEN: ID plain -> ID.x (3/4/5/7)
    if brand == "VOLKSWAGEN":
        m = re.search(r'\bID\s*[.\-]?(3|4|5|7)\b', up)
        if m:
            fam = f"ID.{m.group(1)}"
            if fam in wl3.get(brand, set()):
                disp = disp3.get((brand, fam), fam)
                return _canonical_alias(brand, disp)

    return None

def normaliza_modelo_fusion(marca: str, modelo: str,
                            wl3: Dict[str, Set[str]], disp3: Dict[Tuple[str,str], str], wl2: Dict[str,Set[str]],
                            weights: Dict[str,Dict[str,float]]) -> Tuple[Optional[str], Optional[str], float]:
    brand, model_in_nr, pool, v2_cand, fam_hint = _candidate_pool(marca, modelo, wl3, disp3, wl2)

    hard = _hard_patches(brand, model_in_nr, wl3, disp3)
    if hard:
        return hard, None, 1.0

    if brand == "MERCEDES BENZ":
        m0 = re.match(r'^([ABCES])[-\s]?\d{2,3}', model_in_nr, flags=re.I)
        if m0:
            fam = f"CLASE {m0.group(1).upper()}"
            coupe_present = bool(re.search(r'\bCOUPE\b', model_in_nr))
            fam_coupe = f"{fam} COUPE" if coupe_present else None
            wl3_norm = { _norm(c) for c in wl3.get(brand, set()) }
            chosen = None
            if fam_coupe and _norm(fam_coupe) in wl3_norm:
                chosen = fam_coupe
            elif _norm(fam) in wl3_norm:
                chosen = fam
            if chosen:
                disp_fam = disp3.get((brand, _norm(chosen)), chosen)
                disp_fam = _canonical_alias(brand, disp_fam)
                return disp_fam, None, 1.0

    if not pool:
        return None, None, 0.0
    w = _brand_weights(weights, brand)

    rows = []
    for c in pool:
        feats = _score_features(brand, model_in_nr, c, v2_cand, fam_hint, set(pool))
        s = _linear_score(feats, w)
        if brand == "MERCEDES BENZ" and feats.get("present", 0.0) == 0.0:
            s -= 10.0
        rows.append((c, s))
    rows.sort(key=lambda x: x[1], reverse=True)

    best, s1 = rows[0]
    second, s2 = (rows[1] if len(rows)>1 else (None, 0.0))

    disp_best = disp3.get((brand, best), best)
    disp_best = _canonical_alias(brand, disp_best)
    disp_second = second
    if disp_second:
        disp_second = disp3.get((brand, disp_second), disp_second)
        disp_second = _canonical_alias(brand, disp_second)

    confidence = float(max(0.0, s1 - s2))
    return disp_best, disp_second, confidence

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

def parse_args(argv):
    ap = argparse.ArgumentParser(description="Normalizador fusión v2+v3 (producción) [FULL PATCHED]")
    ap.add_argument("positional_input", nargs="?", help="(compat) input si no usas --input")
    ap.add_argument("--input", help="Ruta de entrada (.xlsx/.xls/.csv/.parquet)")
    ap.add_argument("--sheet", default=None, help="Hoja si Excel")
    ap.add_argument("--make-col", default="make", help="Columna marca preferente (default: make)")
    ap.add_argument("--model-col", default="model", help="Columna modelo preferente (default: model)")
    ap.add_argument("--whitelist", required=True, help="Ruta XLSX whitelist")
    ap.add_argument("--output_excel", help="(compat) salida .xlsx")
    ap.add_argument("--out", help="Salida si no usas --output_excel (admite .xlsx/.csv/.parquet)")
    ap.add_argument("--weights", default="weights.json", help="Pesos JSON (opcional)")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    input_path = args.input or args.positional_input
    if not input_path:
        raise SystemExit("Debes indicar un input: positional o --input")
    out_path = args.output_excel or args.out or input_path

    df = _load_input(input_path, sheet=getattr(args, "sheet", None))

    cols_lower = {c.lower(): c for c in df.columns}
    make_col  = args.make_col if args.make_col in df.columns else (cols_lower.get("make") or cols_lower.get("marca") or cols_lower.get("make_clean"))
    model_col = args.model_col if args.model_col in df.columns else (cols_lower.get("model") or cols_lower.get("modelo"))
    if not make_col or not model_col:
        raise SystemExit("No encuentro columnas de marca/modelo. Usa --make-col/--model-col.")

    wl3 = v3.carga_whitelist(args.whitelist)
    disp3 = v3._v3_build_display_map(args.whitelist)
    try:
        wl2  = v2.carga_whitelist(args.whitelist)
    except Exception:
        wl2 = {}
    weights = _load_weights(args.weights)

    out_best = []
    for _, r in df.iterrows():
        mk = r.get(make_col)
        md = r.get(model_col)
        try:
            best, _, _ = normaliza_modelo_fusion(mk, md, wl3, disp3, wl2, weights)
        except Exception:
            best = None
        out_best.append(best)

    out = df.copy()
    out["modelo_base"] = out_best
    out["make_clean"]  = out[make_col].apply(v3.normalize_brand)
    _save_output(out, out_path)

if __name__ == "__main__":
    main()
