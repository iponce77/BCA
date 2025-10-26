#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
economics_merge_transport_enriched.py
Versión IN-PLACE: escribe el resultado en el MISMO Excel de entrada (sin *_enriched.xlsx).

- País canónico por diccionario
- Categoría por transport_body_map (regex)
- Matching de compound con puntuación (marca + solape tokens + ZIP)
- Precio directo por categoría y fallback (media país+categoría)
- Campos extra: transport_confidence, *_fallback, dias_transporte

Uso recomendado:
  python economics_merge_transport_enriched.py \
    --excel fichas_vehiculos_YYYYMMDD.xlsx \
    --tasas bca_tasas_tarifas_estructuradas.csv \
    --transporte "bca_transporte_estructurado (1).csv" \
    --fijos bca_otros_gastos_es.csv \
    --dest Spain \
    --country_dict country_dict_updated2_extended.csv \
    --body_rules transport_body_map.v2.json \
    --brands compound_brands.v2.json \
    --stops compound_stops.v2.json
"""

import argparse, json, re, unicodedata
from pathlib import Path
import pandas as pd
import json as _json_inj
from pathlib import Path as _PInj
import numpy as np

DEST_ALIASES = {"spain","españa","espana"}

# ----------------- FX -----------------
FX_RATES = {
    "EUR": 1.0,
    "DKK": 0.1340,
    "SEK": 0.08935,
    "PLN": 0.2310,
    "HUF": 0.0026
}

# ----------------- utils -----------------
def nfkd_lower(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\\s+", " ", s)
    return s

def normalize(s: str) -> str:
    return nfkd_lower(s)

def tokenize(s: str):
    return [t for t in re.split(r"[^\w]+", s) if t]

# ----------------- tasas -----------------
def sanitize_tasa_unidad(x: str):
    if pd.isna(x):
        return x
    s = str(x).upper()
    return "%" if "%" in s else "EUR"



def _to_float(x):
    try: return float(x)
    except Exception: return None
def load_tasas_any_ext(path_tasas: str) -> pd.DataFrame:
    ext = str(path_tasas).lower().split(".")[-1]
    if ext in ("xlsx","xls"):
        df = pd.read_excel(path_tasas, sheet_name=0)
    else:
        df = pd.read_csv(path_tasas)
    for col in ["tasa_valor_eur","importe_minimo_eur","precio_min","precio_max","tasa_valor","importe_minimo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    def _san_unidad(x):
        if pd.isna(x): return None
        s = str(x).upper()
        return "%" if "%" in s else "EUR"
    moneda_col = "moneda" if "moneda" in df.columns else ("moneda_origen_valor" if "moneda_origen_valor" in df.columns else None)
    if "tasa_valor_eur" not in df.columns and "tasa_valor" in df.columns:
        if moneda_col:
            df["tasa_valor_eur"] = df.apply(lambda r: (_to_float(r["tasa_valor"]) or 0.0) * FX_RATES_TASAS.get(str(r[moneda_col]).upper(), 1.0), axis=1)
        else:
            df["tasa_valor_eur"] = df["tasa_valor"]
    if "importe_minimo_eur" not in df.columns and "importe_minimo" in df.columns:
        if moneda_col:
            df["importe_minimo_eur"] = df.apply(lambda r: (_to_float(r["importe_minimo"]) or 0.0) * FX_RATES_TASAS.get(str(r[moneda_col]).upper(), 1.0), axis=1)
        else:
            df["importe_minimo_eur"] = df["importe_minimo"]
    df["pais"] = df["pais"].fillna("")
    df["categoria"] = df.get("categoria","").fillna("")
    df["subasta_o_tipo"] = df.get("subasta_o_tipo","").fillna("")
    df["scope"] = df["pais"].map(lambda s: "__GLOBAL__" if nfkd_lower(s) in ("europa","global","todos","__global__") else nfkd_lower(s))
    df["categoria_norm"] = df["categoria"].map(nfkd_lower)
    df["subasta_o_tipo_norm"] = df["subasta_o_tipo"].map(nfkd_lower)
    df["tasa_unidad"] = df.get("tasa_unidad","").map(_san_unidad)
    df["_orden"] = list(range(len(df)))
    return df
_AUCTION_ALIASES = None
_FUEL_ALIASES = None
def _lazy_alias():
    global _AUCTION_ALIASES, _FUEL_ALIASES
    if _AUCTION_ALIASES is None:
        for p in [_PInj.cwd()/ "auction_aliases.json", _PInj(__file__).parent / "auction_aliases.json"]:
            if p.exists():
                try:
                    _AUCTION_ALIASES = _json_inj.load(open(p,"r",encoding="utf-8"))
                    break
                except Exception: pass
        if _AUCTION_ALIASES is None: _AUCTION_ALIASES = {}
    if _FUEL_ALIASES is None:
        for p in [_PInj.cwd()/ "fuel_aliases.json", _PInj(__file__).parent / "fuel_aliases.json"]:
            if p.exists():
                try:
                    _FUEL_ALIASES = _json_inj.load(open(p,"r",encoding="utf-8"))
                    break
                except Exception: pass
        if _FUEL_ALIASES is None: _FUEL_ALIASES = {"electric":{"canonical":"electric","is_electric":True}}
def _is_ev(row: dict) -> bool:
    s = nfkd_lower(row.get("fuel_type") or row.get("Fuel Type") or row.get("fuel") or row.get("Combustible") or "")
    for k,v in _FUEL_ALIASES.items():
        if k in s: return bool(v.get("is_electric",False))
    return any(t in s for t in ["electric","bev"])
def _auction_canon(raw: str):
    s = nfkd_lower(raw or "")
    if s in _AUCTION_ALIASES: return _AUCTION_ALIASES[s], 1.0, "alias_exact"
    if "tesla" in s: return "Tesla", 0.8, "token_subset"
    if "bev" in s: return "BEV", 0.8, "token_subset"
    if any(t in s for t in ["damage","dañado","panne"]): return "Dañados", 0.7, "synonym"
    return "Otros", 0.6, "fallback_otros"
def _pick_band(df_scope: pd.DataFrame, base_eur: float):
    cand = []
    for _,r in df_scope.iterrows():
        pmin, pmax = r.get("precio_min"), r.get("precio_max")
        ok = (pd.isna(pmin) or base_eur >= float(pmin)) and (pd.isna(pmax) or base_eur <= float(pmax))
        if ok: cand.append(r)
    if not cand: return None
    cand = sorted(cand, key=lambda rr: (0 if (pd.isna(rr.get("precio_min")) and pd.isna(rr.get("precio_max"))) else 1, rr.get("_orden",0)))
    return cand[0]
def _component(comp: str, df_tasas: pd.DataFrame, scope: str, auc_canon: str, base_eur: float, is_ev: bool):
    df_scope = df_tasas[(df_tasas["scope"]==scope) & (df_tasas["categoria_norm"]==comp)]
    if df_scope.empty: return 0.0, "no_scope_match", 0.0
    if comp=="diagnostico" and not is_ev: return 0.0, "skip_not_ev", 0.0
    sub = df_scope[df_scope["subasta_o_tipo_norm"]==nfkd_lower(auc_canon)]
    if sub.empty: sub = df_scope[df_scope["subasta_o_tipo_norm"]=="otros"]
    if sub.empty: sub = df_scope
    row = _pick_band(sub, base_eur)
    if row is None: return 0.0, "no_price_band", 0.0
    unidad = row.get("tasa_unidad")
    fee = 0.0
    if unidad == "%":
        fee = (float(row.get("tasa_valor_eur") or 0.0)/100.0)*float(base_eur or 0.0)
        mn = row.get("importe_minimo_eur")
        if pd.notna(mn): fee = max(fee, float(mn))
    elif unidad == "EUR":
        fee = float(row.get("tasa_valor_eur") or 0.0)
    rule = f"comp={comp}|scope={scope}|pais={row.get('pais')}|subasta={row.get('subasta_o_tipo')}|unidad={unidad}|val={row.get('tasa_valor_eur')}|min={row.get('importe_minimo_eur')}|pmin={row.get('precio_min')}|pmax={row.get('precio_max')}"
    return float(fee), rule, 0.7
def compute_commission_bundle(row: dict, df_tasas: pd.DataFrame):
    _lazy_alias()
    base_eur = _to_float(row.get("vehicle_base_price_eur") or row.get("winning_bid") or 0.0)
    country = nfkd_lower(row.get("sale_info_country") or row.get("saleCountry") or row.get("sale_country") or "")
    auction_raw = row.get("auction_name") or row.get("BCA Sale Name") or row.get("sale_name") or row.get("bca_sale_name") or row.get("auction") or ""
    auc_canon, a_conf, a_m = _auction_canon(auction_raw)
    is_ev = _is_ev(row)
    scopes = [country, "__GLOBAL__"]
    if auc_canon=="BEV" and country!="dinamarca": scopes = ["__GLOBAL__"]
    comps = {}; total=0.0
    for comp in ["adquisicion","exportacion","documentos","tramitacion","diagnostico"]:
        fee, rule, conf = 0.0, "no_scope_match", 0.0
        for sc in scopes:
            fee, rule, conf = _component(comp, df_tasas, sc, auc_canon, base_eur or 0.0, is_ev)
            if fee>0.0 or comp!="adquisicion": break
        comps[comp] = {"fee":fee,"rule":rule,"conf":conf}; total += fee
    err = None
    if comps["adquisicion"]["fee"] <= 0.0: err = "sin_adquisicion_valida"
    audit = {"country": country, "auction_raw": auction_raw, "auction_canonical": auc_canon, "auction_confidence": a_conf, "auction_method": a_m, "is_electric": is_ev, "components": comps}
    return {"commission_acquisition_eur": comps["adquisicion"]["fee"], "commission_export_eur": comps["exportacion"]["fee"],
            "commission_docs_eur": comps["documentos"]["fee"], "commission_admin_eur": comps["tramitacion"]["fee"],
            "commission_diagnostic_eur": comps["diagnostico"]["fee"], "commission_total_eur": total,
            "commission_rules_json": _json_inj.dumps(audit, ensure_ascii=False), "commission_error": err}
def compute_commission(row: dict, df_tasas: pd.DataFrame, base_price_eur: float):
    out = compute_commission_bundle(row, df_tasas)
    return float(out["commission_total_eur"] or 0.0), out["commission_rules_json"]
# =================== FIN DE LA NUEVA LOGICA DE TASAS ===================


    sub = df_tasas[df_tasas["pais"].astype(str).str.lower() == str(pais).lower()]
    if sub.empty:
        return 0.0, "sin_reglas_pais"

    auction = normalize(row.get("auction_name") or "")
    if auction:
        sub_by_auction = sub[sub["subasta_o_tipo"].fillna("").astype(str).str.lower() == auction]
        if not sub_by_auction.empty:
            sub = sub_by_auction

    best_val, best_note = 0.0, ""
    for _, r in sub.iterrows():
        pmin, pmax = r.get("precio_min"), r.get("precio_max")
        if pd.notna(pmin) and (base_price_eur < float(pmin)):
            continue
        if pd.notna(pmax) and (base_price_eur > float(pmax)):
            continue

        unidad = r.get("tasa_unidad")
        if unidad == "%":
            fee = (float(r["tasa_valor_eur"]) / 100.0) * base_price_eur
            min_eur = r.get("importe_minimo_eur")
            if pd.notna(min_eur):
                fee = max(fee, float(min_eur))
        elif unidad == "EUR":
            fee = float(r.get("tasa_valor_eur") or 0.0)
        else:
            continue

        if fee > best_val:
            best_val = fee
            best_note = r.get("subasta_o_tipo") or "regla_pais"

    return float(best_val), best_note or "regla_pais"

# ----------------- FX base price -----------------
def is_sold(row: dict) -> bool:
    val = row.get("lot_status")
    return isinstance(val, str) and val.strip().lower() == "sold"

def winning_bid_eur_from_row(row: dict):
    if not is_sold(row):
        return None
    wb = row.get("winning_bid")
    if pd.isna(wb):
        return None
    try:
        wb = float(wb)
    except Exception:
        return None
    curr = str(row.get("currency") or "EUR").upper()
    rate = FX_RATES.get(curr, 1.0)
    return wb * rate

# ----------------- country dict & body rules -----------------
def load_country_dict(path_csv: str):
    cdf = pd.read_csv(path_csv)
    if not {"variant","country_canonical"}.issubset(cdf.columns):
        raise ValueError("country_dict requiere columnas 'variant' y 'country_canonical'")
    cdf["_variant_norm"] = cdf["variant"].map(nfkd_lower)
    cdf["_canon_norm"] = cdf["country_canonical"].map(nfkd_lower)
    return dict(zip(cdf["_variant_norm"], cdf["_canon_norm"]))

def canon_country(val, cmap: dict):
    v = nfkd_lower(val)
    return cmap.get(v, v)

def load_body_rules(path_json):
    cfg = json.load(open(path_json, "r", encoding="utf-8"))
    rules = cfg.get("rules", [])
    compiled = []
    for r in rules:
        if r.get("match_type") == "regex":
            pats = [re.compile(pat, re.IGNORECASE) for pat in r.get("patterns", [])]
        else:
            pats = []
        compiled.append((r.get("category"), r.get("match_type"), pats))
    return compiled

def map_category(text_a, text_b, compiled_rules):
    blob = nfkd_lower((text_a or "") + " " + (text_b or ""))
    for cat, mtype, pats in compiled_rules:
        if mtype == "regex" and any(p.search(blob) for p in pats):
            return cat
    for cat, mtype, pats in compiled_rules:
        if mtype == "fallback":
            return cat
    return "passenger"

# ----------------- compound matching & fallback -----------------
def score_compound(ubic_norm: str, cand_row: dict, brand_set: set, stop_set: set) -> int:
    oc = nfkd_lower(cand_row.get("origen_compound",""))
    u_toks = set(t for t in tokenize(ubic_norm) if t and not t.isdigit())
    oc_toks = set(t for t in tokenize(oc) if t and not t.isdigit())
    brand_score = sum(3 for b in brand_set if (b in ubic_norm and b in oc))
    overlap = len([t for t in (u_toks & oc_toks) if t not in stop_set])
    zip_bonus = 0
    # ZIP bonus si hay match exacto con destino_zip
    try:
        zips = [t for t in tokenize(ubic_norm) if t.isdigit()]
        if zips and ("destino_zip" in cand_row):
            z = int(zips[0])
            if int(cand_row.get("destino_zip") or -1) == z:
                zip_bonus = 2
    except Exception:
        pass
    return int(brand_score + overlap + zip_bonus)

def select_transport_price(cat: str, cand_row: dict):
    if cat == "suv":
        return cand_row.get("precio_suv_eur")
    if cat == "lcv":
        return cand_row.get("precio_lcv1_eur") or cand_row.get("precio_lcv2_eur")
    return cand_row.get("precio_passenger_car_eur")

def build_fallback_table(trans_df: pd.DataFrame, country_map: dict) -> pd.DataFrame:
    t = trans_df.copy()
    t["_po_canon"] = t["pais_origen"].map(lambda x: canon_country(x, country_map))
    out = []
    for cat, col in [("passenger","precio_passenger_car_eur"),
                     ("suv","precio_suv_eur"),
                     ("lcv","precio_lcv1_eur")]:
        g = t.groupby("_po_canon")[col].mean(numeric_only=True).reset_index()
        g["vehicle_category"] = cat
        g = g.rename(columns={col:"avg_price_eur", "_po_canon":"origin_country_canon"})
        g["n_routes"] = t.groupby("_po_canon")[col].count().reindex(g["origin_country_canon"]).values
        out.append(g)
    return pd.concat(out, ignore_index=True)

def compute_transport_enriched(row: dict,
                               trans_df: pd.DataFrame,
                               dest_country: str,
                               country_map: dict,
                               compiled_body_rules,
                               brand_set: set,
                               stop_set: set,
                               fallback_tbl: pd.DataFrame):
    origin_raw = row.get("sale_info_country") or row.get("saleCountry") or row.get("sale_country") or ""
    origin_canon = canon_country(origin_raw, country_map)
    dest_canon = canon_country(dest_country, country_map)

    t = trans_df.copy()
    t["_po_canon"] = t["pais_origen"].map(lambda x: canon_country(x, country_map))
    t["_pd_canon"] = t["pais_destino"].map(lambda x: canon_country(x, country_map))
    cand = t[(t["_po_canon"] == origin_canon) & (t["_pd_canon"] == dest_canon)]

    # Si no hay rutas para ese país->destino, caer a fallback puro
    if cand.empty:
        cat = map_category(row.get("Tipo Carrocería"), row.get("Tipo de vehículo"), compiled_body_rules)
        ft = fallback_tbl[(fallback_tbl["origin_country_canon"]==origin_canon) & (fallback_tbl["vehicle_category"]==cat)]
        if not ft.empty and float(ft.iloc[0]["avg_price_eur"] or 0) > 0:
            return 0.0, "sin_tarifa_transporte", 0.5, float(ft.iloc[0]["avg_price_eur"]), f"avg_price_fallback:{origin_canon}|{cat}|n={int(ft.iloc[0]['n_routes'])}", 0.5, None
        return 0.0, "sin_tarifa_transporte", 0.5, 0.0, "", 0.0, None

    # Matching por score
    ubic_norm = nfkd_lower(row.get("Ubicación") or row.get("ubicacion") or "")
    cand = cand.copy()
    cand["match_score"] = cand.apply(lambda rr: score_compound(ubic_norm, rr, brand_set, stop_set), axis=1)
    best = cand.sort_values("match_score", ascending=False).iloc[0].to_dict()

    # Precio directo por categoría
    cat = map_category(row.get("Tipo Carrocería"), row.get("Tipo de vehículo"), compiled_body_rules)
    price_direct = float(select_transport_price(cat, best) or 0.0)
    rule = f"{best.get('origen_compound')}|score={int(best.get('match_score') or 0)}"
    conf = 0.9 if best["match_score"] >= 3 else (0.7 if best["match_score"] >= 1 else 0.5)
    dias = best.get("dias_full_truck")

    if price_direct > 0:
        return price_direct, rule, conf, 0.0, "", 0.0, dias

    # Fallback si precio directo = 0
    ft = fallback_tbl[(fallback_tbl["origin_country_canon"]==origin_canon) & (fallback_tbl["vehicle_category"]==cat)]
    if not ft.empty and float(ft.iloc[0]["avg_price_eur"] or 0) > 0:
        fb_price = float(ft.iloc[0]["avg_price_eur"])
        fb_rule = f"avg_price_fallback:{origin_canon}|{cat}|n={int(ft.iloc[0]['n_routes'])}"
        return 0.0, rule, conf, fb_price, fb_rule, 0.5, dias

    return 0.0, rule, conf, 0.0, "", 0.0, dias

# ----------------- fijos ES -----------------
def sum_fixed_es(df_fijos: pd.DataFrame) -> float:
    if df_fijos is None or df_fijos.empty or "valor_eur" not in df_fijos.columns:
        return 0.0
    return float(df_fijos["valor_eur"].fillna(0).sum())

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--tasas", required=False, default=None)
    ap.add_argument("--transporte", required=True)
    ap.add_argument("--fijos", required=False, default=None)
    ap.add_argument("--dest", default="Spain")
    # nuevos insumos para transporte enriquecido
    ap.add_argument("--country_dict", default="country_dict_updated2_extended.csv")
    ap.add_argument("--body_rules",   default="transport_body_map.v2.json")
    ap.add_argument("--brands",       default="compound_brands.v2.json")
    ap.add_argument("--stops",        default="compound_stops.v2.json")

    args = ap.parse_args()

    excel_path = Path(args.excel)
    df = pd.read_excel(excel_path, dtype=str)

    # Filtrar Sold
    df = df[df["lot_status"].fillna("").str.lower().eq("sold")].copy()

    # --- FX base price ---
    df["vehicle_base_price_eur"] = [winning_bid_eur_from_row(r) for _, r in df.iterrows()]

    # --- Tasas (si se provee) ---
    if args.tasas:
        tasas = load_tasas_any_ext(args.tasas)
    else:
        tasas = pd.DataFrame(columns=["pais","subasta_o_tipo","tasa_unidad","tasa_valor_eur","importe_minimo_eur","precio_min","precio_max"])

    comm_vals, comm_notes = [], []
    for _, r in df.iterrows():
        base_eur = r.get("vehicle_base_price_eur")
        base_eur = float(base_eur) if base_eur not in (None, "", "nan") else None
        cval, cnote = compute_commission(r, tasas, base_eur or 0.0)
        comm_vals.append(cval)
        comm_notes.append(cnote)
    df["bca_commission_eur"]  = comm_vals
    df["bca_commission_rule"] = comm_notes

    # --- Transporte enriquecido ---
    df_trans = pd.read_csv(args.transporte)
    country_map = load_country_dict(args.country_dict)
    compiled_body = load_body_rules(args.body_rules)
    brand_set = set(json.load(open(args.brands, encoding="utf-8")))
    stop_set = set(json.load(open(args.stops, encoding="utf-8")))
    fallback_tbl = build_fallback_table(df_trans, country_map)

    # auditoría auxiliares
    df["origin_country_canon"] = df["sale_info_country"].combine_first(df.get("saleCountry")).combine_first(df.get("sale_country")).map(lambda x: canon_country(x, country_map))
    df["dest_country_canon"] = canon_country(args.dest, country_map)
    df["vehicle_category"] = [map_category(a,b, compiled_body) for a,b in zip(df.get("Tipo Carrocería"), df.get("Tipo de vehículo"))]

    tvals, tnotes, tconfs = [], [], []
    fb_vals, fb_rules, fb_confs = [], [], []
    dias_vals = []
    for _, r in df.iterrows():
        tval, tnote, tconf, fb_val, fb_rule, fb_conf, dias = compute_transport_enriched(
            r, df_trans, args.dest, country_map, compiled_body, brand_set, stop_set, fallback_tbl
        )
        tvals.append(tval); tnotes.append(tnote); tconfs.append(tconf)
        fb_vals.append(fb_val); fb_rules.append(fb_rule); fb_confs.append(fb_conf)
        dias_vals.append(dias)

    df["transport_eur"] = tvals
    df["transport_rule"] = tnotes
    df["transport_confidence"] = tconfs
    df["transport_eur_fallback"] = fb_vals
    df["transport_rule_fallback"] = fb_rules
    df["transport_confidence_fallback"] = fb_confs
    df["dias_transporte"] = dias_vals

    # --- Fijos ES (si se provee) ---
    if args.fijos:
        df_fijos = pd.read_csv(args.fijos)
        fijos_total = sum_fixed_es(df_fijos)
    else:
        fijos_total = 0.0

    # --- Económico total (suma fallback si directo=0) ---
    final_transport = np.where((pd.to_numeric(df["transport_eur"], errors="coerce").fillna(0) > 0),
                               pd.to_numeric(df["transport_eur"], errors="coerce").fillna(0),
                               pd.to_numeric(df["transport_eur_fallback"], errors="coerce").fillna(0))
    df["economic_total_eur"] = pd.to_numeric(df["vehicle_base_price_eur"], errors="coerce").fillna(0) + \
                               pd.to_numeric(df["bca_commission_eur"], errors="coerce").fillna(0) + \
                               final_transport + float(fijos_total or 0.0)

    # --- Guardar IN-PLACE (mismo nombre de entrada) ---
    df.to_excel(excel_path, index=False)
    print(f"OK: actualizado in-place {excel_path}")

if __name__ == "__main__":
    main()
