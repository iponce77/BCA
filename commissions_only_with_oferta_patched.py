# commissions_only_with_oferta.py
# -*- coding: utf-8 -*-
"""
Cálculo de comisiones BCA + Export de OFERTA (SIN filtrar NOT SOLD)
-------------------------------------------------------------------
- Genera un CSV de oferta a partir del Excel completo (sold y not sold).
- Calcula comisiones SOLO para vendidos; deja NaN en not sold.
- No elimina filas del Excel original; escribe salida enriquecida.
"""

import re
import argparse
import json
import math
import unicodedata
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Utilidades generales
# =========================
def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def to_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        v = float(str(x).replace(",", "."))
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def load_country_dict(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if 'variant' in cols and 'country_canonical' in cols:
        src, dst = cols['variant'], cols['country_canonical']
    elif 'raw' in cols and 'canon' in cols:
        src, dst = cols['raw'], cols['canon']
    else:
        cs = list(df.columns)
        if len(cs) < 2:
            raise ValueError('country_dict.csv con columnas insuficientes.')
        src, dst = cs[0], cs[1]
    d: Dict[str, str] = {}
    for _, r in df.iterrows():
        d[normalize_text(r[src])] = str(r[dst]).strip()
    return d

def load_location_aliases(path: Optional[Path]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    if not path or not Path(path).exists():
        return d
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    head_col = cols.get('location_head') or list(df.columns)[0]
    country_col = cols.get('country_canonical') or (list(df.columns)[1] if len(df.columns) > 1 else head_col)
    for _, r in df.iterrows():
        d[normalize_text(str(r[head_col]))] = str(r[country_col]).strip()
    return d

def extract_location_head(ubic: str) -> str:
    """
    Devuelve el 'head' de Ubicación (parte izquierda antes de un separador),
    normalizando guiones Unicode y espacios. Ejemplos:
        "Barcelona – PEEP" -> "Barcelona"
        "Madrid - BuyerGateway" -> "Madrid"
    """
    s = str(ubic or "").strip()
    # Normaliza varios tipos de guiones a '-'
    s = s.replace("–", "-").replace("—", "-").replace("-", "-")
    # Divide una sola vez por " - " con espacios flexibles
    parts = re.split(r"\s+-\s+", s, maxsplit=1)
    head = parts[0].strip() if parts else s
    # Quita restos de puntuación común
    head = head.strip(" -–—:|•·")
    # En algunos ficheros viene "Ciudad, País" -> nos quedamos con la ciudad
    head = head.split(",")[0].strip()
    return head

def canon_country(name: str, cdict: Dict[str, str]) -> str:
    key = normalize_text(name)
    return str(cdict.get(key, name)).strip()


def load_json(path: Optional[Path], default):
    if not path or not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def as_bool_sold(val: Any) -> bool:
    v = normalize_text(str(val))
    return v == "sold"


# FX simplificado (ajusta si precisas)
FX = {
    "eur": 1.0,
    "dkk": 0.134,
    "sek": 0.088,
    "pln": 0.23,
    "huf": 0.0026,
}


def fx_eur(amount: Optional[float], currency: Optional[str]) -> Optional[float]:
    if amount is None:
        return None
    code = normalize_text(currency) or "eur"
    rate = FX.get(code, 1.0)
    return amount * rate


# =========================
# Carga de tasas
# =========================
def load_tasas(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    cols = {normalize_text(c): c for c in df.columns}

    def col_like(*names) -> Optional[str]:
        for n in names:
            if n in cols:
                return cols[n]
        return None

    mapping = {
        "pais": col_like("pais", "country", "scope"),
        "categoria": col_like("categoria", "category", "tipo"),
        "subasta_o_tipo": col_like("subasta_o_tipo", "subasta", "venta", "sale", "auction"),
        "precio_min": col_like("precio_min", "min", "desde"),
        "precio_max": col_like("precio_max", "max", "hasta"),
        "tasa_unidad": col_like("tasa_unidad", "unidad", "unit"),
        "tasa_valor": col_like("tasa_valor", "valor", "value"),
        "importe_minimo": col_like("importe_minimo", "minimo", "minimum"),
        "moneda": col_like("moneda", "currency"),
        "moneda_origen_valor": col_like("moneda_origen_valor", "currency_valor", "currency_value"),
    }
    for k, v in mapping.items():
        if v is None and k in ["importe_minimo", "moneda", "moneda_origen_valor"]:
            continue
        if v is None:
            raise ValueError(f"Falta columna requerida en tasas: {k}")

    out = pd.DataFrame()
    for k, v in mapping.items():
        out[k] = df[v] if v in df.columns else np.nan

    # Normaliza numéricos
    for c in ["precio_min", "precio_max", "tasa_valor", "importe_minimo"]:
        out[c] = out[c].apply(to_float)

    # Normaliza enums
    out["pais_norm"] = out["pais"].apply(lambda x: normalize_text(str(x)))
    out["categoria_norm"] = out["categoria"].apply(lambda x: normalize_text(str(x)))
    out["subasta_norm"] = out["subasta_o_tipo"].apply(lambda x: normalize_text(str(x)))
    out["tasa_unidad_norm"] = out["tasa_unidad"].apply(lambda x: normalize_text(str(x)))
    out["moneda_norm"] = out["moneda"].apply(lambda x: normalize_text(str(x)))
    out["moneda_origen_valor_norm"] = out["moneda_origen_valor"].apply(lambda x: normalize_text(str(x)))

    return out


# =========================
# Selección de regla/banda
# =========================
CAT_ORDER = ["adquisicion", "exportacion", "documentos", "tramitacion", "diagnostico"]

def pick_band(rows: pd.DataFrame, base_price: float) -> Optional[pd.Series]:
    cand = rows.copy()
    in_band = cand[
        (cand["precio_min"].apply(lambda v: v is None or base_price >= v)) &
        (cand["precio_max"].apply(lambda v: v is None or base_price <= v))
    ]
    if not in_band.empty:
        in_band = in_band.assign(
            span=in_band.apply(
                lambda r: (float("inf") if r["precio_min"] is None or r["precio_max"] is None
                           else (r["precio_max"] - r["precio_min"])), axis=1
            ),
            minv=in_band["precio_min"].apply(lambda v: v if v is not None else -1e18)
        ).sort_values(["span", "minv"], ascending=[True, True])
        return in_band.iloc[0]

    below = cand[cand["precio_min"].apply(lambda v: v is not None and v <= base_price)]
    if not below.empty:
        return below.sort_values("precio_min", ascending=False).iloc[0]

    above = cand[cand["precio_min"].apply(lambda v: v is not None and v > base_price)]
    if not above.empty:
        return above.sort_values("precio_min", ascending=True).iloc[0]

    noband = cand[(cand["precio_min"].isna()) & (cand["precio_max"].isna())]
    if not noband.empty:
        return noband.iloc[0]

    if not cand.empty:
        return cand.iloc[0]
    return None


def compute_amount_eur(rule_row: pd.Series, base_price: float) -> float:
    unit = normalize_text(rule_row.get("tasa_unidad_norm"))
    valor = to_float(rule_row.get("tasa_valor"))
    minimo = to_float(rule_row.get("importe_minimo"))
    moneda_valor = normalize_text(rule_row.get("moneda_origen_valor_norm") or rule_row.get("moneda_norm"))

    if unit in {"%", "porcentaje", "percent"}:
        amount = (valor or 0.0) * base_price / 100.0
        if minimo is not None:
            amount = max(amount, minimo)
    else:
        amount = (valor or 0.0)
        if moneda_valor and moneda_valor != "eur":
            amount = fx_eur(amount, moneda_valor)

    return float(amount or 0.0)


def rule_string(rule_row: pd.Series, categoria: str, pais_scope: str, subasta: str) -> str:
    pm = rule_row.get("precio_min")
    px = rule_row.get("precio_max")
    unit = rule_row.get("tasa_unidad")
    val = rule_row.get("tasa_valor")
    minv = rule_row.get("importe_minimo")
    curv = rule_row.get("moneda_origen_valor") or rule_row.get("moneda")
    return (
        f"cat={categoria}|scope={pais_scope}|subasta={subasta}|"
        f"band=[{pm},{px}]|unit={unit}|value={val}|min={minv}|currency_val={curv}"
    )


def select_rule(
    tasas: pd.DataFrame,
    categoria: str,
    pais_canon: str,
    auction_canon: str,
    base_price: float
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    cat = normalize_text(categoria)
    if cat not in CAT_ORDER:
        if cat in {"compra", "buyer", "buy", "fee"}:
            cat = "adquisicion"

    def scope_rows(scope_country: Optional[str]) -> pd.DataFrame:
        if scope_country is None:
            mask = tasas["pais_norm"].isin(["europa", "europe", "todos", "all", "global", ""])
        else:
            mask = tasas["pais_norm"] == normalize_text(scope_country)
        return tasas[mask & (tasas["categoria_norm"] == cat)].copy()

    if cat in {"exportacion", "documentos", "diagnostico", "tramitacion"}:
        rows = scope_rows(pais_canon)
        if not rows.empty:
            wild = rows[rows["subasta_norm"].isin(["otros", "other", "", None])]
            pool = wild if not wild.empty else rows
            r = pick_band(pool, base_price)
            if r is not None:
                return compute_amount_eur(r, base_price), rule_string(r, cat, pais_canon, "otros"), None

        rows = scope_rows(None)
        if not rows.empty:
            wild = rows[rows["subasta_norm"].isin(["otros", "other", "", None])]
            pool = wild if not wild.empty else rows
            r = pick_band(pool, base_price)
            if r is not None:
                return compute_amount_eur(r, base_price), rule_string(r, cat, "global", "otros"), None

        return None, None, f"sin_regla_{cat}"

    rows = scope_rows(pais_canon)
    if not rows.empty:
        exact = rows[rows["subasta_norm"] == normalize_text(auction_canon)]
        pool = exact if not exact.empty else rows[rows["subasta_norm"].isin(["otros", "other", "", None])]
        if pool.empty:
            pool = rows
        r = pick_band(pool, base_price)
        if r is not None:
            return compute_amount_eur(r, base_price), rule_string(r, "adquisicion", pais_canon, auction_canon if not exact.empty else "otros"), None

    rows = scope_rows(None)
    if not rows.empty:
        exact = rows[rows["subasta_norm"] == normalize_text(auction_canon)]
        pool = exact if not exact.empty else rows[rows["subasta_norm"].isin(["otros", "other", "", None])]
        if pool.empty:
            pool = rows
        r = pick_band(pool, base_price)
        if r is not None:
            return compute_amount_eur(r, base_price), rule_string(r, "adquisicion", "global", auction_canon if not exact.empty else "otros"), None

    return None, None, "sin_adquisicion_valida"


# =========================
# Pipeline de una fila
# =========================
def compute_commissions_for_row(
    row: pd.Series,
    tasas: pd.DataFrame,
    country_dict: Dict[str, str],
    auction_aliases: Dict[str, str],
    fuel_aliases: Dict[str, Any],
    include_not_sold: bool,
    location_aliases: Dict[str, str]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "commission_adquisicion_eur": np.nan,
        "commission_exportacion_eur": np.nan,
        "commission_documentos_eur": np.nan,
        "commission_tramitacion_eur": np.nan,
        "commission_diagnostico_eur": np.nan,
        "commission_total_eur": np.nan,
        "commission_rules_json": "",
        "commission_error": "",
    }

    is_sold = as_bool_sold(row.get("lot_status"))
    if include_not_sold:
        out["commission_eligible"] = True
        out["commission_reason"] = "forced_include"
        is_sold = True
    else:
        out["commission_eligible"] = bool(is_sold)
        out["commission_reason"] = "sold" if is_sold else "not_sold_skipped"
        if not is_sold:
            out["bca_commission_eur"] = 0.0
            out["bca_commission_rule"] = "bundle_json"
            return out

    # Precio base
    vehicle_base_price_eur = to_float(row.get("vehicle_base_price_eur"))
    if vehicle_base_price_eur is None or vehicle_base_price_eur <= 0:
        winning_bid = to_float(row.get("winning_bid"))
        currency = row.get("currency") or "EUR"
        base_price = fx_eur(winning_bid, currency)
    else:
        base_price = vehicle_base_price_eur

    if base_price is None:
        out["commission_error"] = "sin_precio_base"
        out["bca_commission_eur"] = np.nan
        out["bca_commission_rule"] = "bundle_json"
        return out

    # País
    raw_country = row.get("sale_info_country") or row.get("saleCountry") or row.get("sale_country") or ""
    pais_canon = canon_country(str(raw_country), country_dict)
    # Inferir país desde Ubicación si genérico
    if normalize_text(pais_canon) in {"eu","ue","europe","europa","","nan","none","global"}:
        ubic = row.get("Ubicación") or row.get("Ubicacion") or row.get("Location") or ""
        head = extract_location_head(ubic)
        key = normalize_text(head)
        if key in location_aliases:
            pais_canon = location_aliases[key]
        elif head:
            pais_canon = canon_country(head, country_dict)

    # Subasta (canónica) - robusto a NaN/float
    raw_auction = (
        row.get("auction_name") or
        row.get("BCA Sale Name") or
        row.get("sale_name") or
        row.get("bca_sale_name") or
        ""
    )
    raw_auction_str = str(raw_auction).strip()
    auction_key = normalize_text(raw_auction_str)
    alias_val = auction_aliases.get(auction_key) if isinstance(auction_aliases, dict) else None
    if alias_val is None or (isinstance(alias_val, str) and alias_val.strip() == ""):
        auction_canon = raw_auction_str
    else:
        auction_canon = str(alias_val).strip()

    # Fuel / EV
    fuel_raw = row.get("fuel_type") or row.get("combustible") or row.get("Fuel Type") or ""
    fuel_key = normalize_text(str(fuel_raw))
    is_electric = False
    if isinstance(fuel_aliases, dict):
        val = fuel_aliases.get(fuel_key)
        if isinstance(val, dict):
            is_electric = bool(val.get("is_electric", False))
        elif isinstance(val, bool):
            is_electric = val
        else:
            is_electric = fuel_key in {"electric", "ev", "bev"}
    else:
        is_electric = fuel_key in {"electric", "ev", "bev"}

    rules = {}
    errors = []

    # ADQUISICIÓN
    amt, rstr, err = select_rule(tasas, "adquisicion", pais_canon, auction_canon, base_price)
    if amt is not None:
        out["commission_adquisicion_eur"] = amt
        rules["adquisicion"] = rstr
    elif err:
        errors.append(err)

    # EXPORTACIÓN
    amt, rstr, err = select_rule(tasas, "exportacion", pais_canon, auction_canon, base_price)
    if amt is not None:
        out["commission_exportacion_eur"] = amt
        rules["exportacion"] = rstr
    elif err:
        errors.append(err)

    # DOCUMENTOS
    amt, rstr, err = select_rule(tasas, "documentos", pais_canon, auction_canon, base_price)
    if amt is not None:
        out["commission_documentos_eur"] = amt
        rules["documentos"] = rstr
    elif err:
        errors.append(err)

    # TRAMITACIÓN (si existe en tabla)
    if (tasas["categoria_norm"] == "tramitacion").any():
        amt, rstr, err = select_rule(tasas, "tramitacion", pais_canon, auction_canon, base_price)
        if amt is not None:
            out["commission_tramitacion_eur"] = amt
            rules["tramitacion"] = rstr
        elif err:
            errors.append(err)

    # DIAGNÓSTICO (solo EV)
    if is_electric and (tasas["categoria_norm"] == "diagnostico").any():
        amt, rstr, err = select_rule(tasas, "diagnostico", pais_canon, auction_canon, base_price)
        if amt is not None:
            out["commission_diagnostico_eur"] = amt
            rules["diagnostico"] = rstr
        elif err:
            errors.append(err)

    # Total
    parts = [
        out["commission_adquisicion_eur"],
        out["commission_exportacion_eur"],
        out["commission_documentos_eur"],
        out["commission_tramitacion_eur"],
        out["commission_diagnostico_eur"],
    ]
    total = sum([p for p in parts if (p is not None and not pd.isna(p))], 0.0)
    out["commission_total_eur"] = float(total)

    # Auditoría
    out["commission_rules_json"] = json.dumps({
        "pais_canonico": pais_canon,
        "auction_canonico": auction_canon,
        "is_electric": is_electric,
        "base_price_eur": base_price,
        "rules": rules,
    }, ensure_ascii=False)

    out["commission_error"] = ";".join(sorted(set(errors))) if errors else ""

    # Compat BCA
    out["bca_commission_eur"] = out["commission_total_eur"]
    out["bca_commission_rule"] = "bundle_json"

    return out


# =========================
# Construcción de OFERTA
# =========================
def build_oferta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una tabla de oferta sin filtrar, con claves normalizadas y deduplicación por VIN si existe.
    """
    # Columnas candidatas
    vin_cols = ["vin", "VIN", "Vin"]
    make_cols = ["make_clean", "make", "Marca"]
    model_cols = ["modelo_base", "model", "model_bca_raw", "model_ganvam_raw", "Modelo"]
    year_cols = ["year", "anio", "Año", "Fecha Matriculación", "registration_date", "first_registration"]
    auction_cols = ["auction_name", "BCA Sale Name", "sale_name", "bca_sale_name"]
    bids_cols = ["number_of_bids", "bids", "bids_count"]
    country_cols = ["sale_info_country", "saleCountry", "sale_country"]

    def first_col(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    vc = first_col(vin_cols)
    mc = first_col(make_cols)
    mdl = first_col(model_cols)
    yc = first_col(year_cols)
    ac = first_col(auction_cols)
    bc = first_col(bids_cols)
    cc = first_col(country_cols)

    out = pd.DataFrame()
    if vc: out["vin"] = df[vc]
    if mc: out["make"] = df[mc]
    if mdl: out["model"] = df[mdl]
    if yc: out["year_raw"] = df[yc]
    if ac: out["auction_name"] = df[ac]
    if bc: out["number_of_bids"] = df[bc]
    if cc: out["sale_info_country"] = df[cc]

    # Normalizaciones útiles
    out["make_key"] = out.get("make", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    out["modelo_base_merge"] = out.get("model", pd.Series(dtype=str)).astype(str).str.strip()
    out["modelo_base_key"] = out["modelo_base_merge"].str.lower()

    # Año: intenta parsear de fecha o texto
    def to_year(v):
        s = str(v)
        # Intenta yyyy-mm-dd o dd/mm/yyyy...
        try:
            dt = pd.to_datetime(v, errors="coerce", dayfirst=True)
            if pd.notna(dt):
                return int(dt.year)
        except Exception:
            pass
        # Intenta coger un 4 dígitos
        import re
        m = re.search(r"(19|20)\d{2}", s)
        return int(m.group(0)) if m else np.nan

    out["year_norm"] = out["year_raw"].apply(to_year) if "year_raw" in out else np.nan

    # Deduplicación por VIN si existe
    if "vin" in out.columns:
        out = out.drop_duplicates(subset=["vin"], keep="first")

    # Orden de columnas
    preferred = ["vin", "make", "model", "year_norm", "sale_info_country", "auction_name", "number_of_bids",
                 "make_key", "modelo_base_merge", "modelo_base_key", "year_raw"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    return out


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Comisiones + Oferta (sin filtrar NOT SOLD)")
    ap.add_argument("--excel", required=True, help="Excel de fichas de entrada")
    ap.add_argument("--tasas", required=True, help="CSV/XLSX con tasas estructuradas")
    ap.add_argument("--auction_aliases", required=False, default="", help="JSON de alias de subastas")
    ap.add_argument("--include-not-sold", action="store_true", help="Calcula comisiones también para NOT SOLD (estimación)")
    ap.add_argument("--location_country_aliases", required=False, default="", help="CSV de alias de location_head -> país canónico")
    ap.add_argument("--fuel_aliases", required=False, default="", help="JSON de alias de combustible (is_electric)")
    ap.add_argument("--country_dict", required=False, default="", help="CSV diccionario de países variant→canonical")
    ap.add_argument("--sheet", required=False, default=None, help="Nombre de hoja si aplica")
    ap.add_argument("--oferta_out", required=False, default="", help="Ruta CSV de oferta (si vacío, auto)")
    ap.add_argument("--out", required=False, default="", help="Excel de salida (si vacío, sobrescribe el de entrada)")
    args = ap.parse_args()

    excel_path = Path(args.excel)
    tasas_path = Path(args.tasas)
    out_path = Path(args.out) if args.out else excel_path
    oferta_path = Path(args.oferta_out) if args.oferta_out else Path(excel_path.parent, f"{excel_path.stem}_oferta.csv")

    # Cargas auxiliares
    country_dict = load_country_dict(Path(args.country_dict)) if args.country_dict else {}

    # Alias de subastas (robusto a NaN/float)
    auction_aliases = load_json(Path(args.auction_aliases), {}) or {}
    include_not_sold = bool(args.include_not_sold)
    location_aliases = load_location_aliases(Path(args.location_country_aliases)) if args.location_country_aliases else {}
    def _is_nan(x):
        return isinstance(x, float) and math.isnan(x)
    if isinstance(auction_aliases, dict):
        auction_aliases = {
            normalize_text(str(k)): ("" if v is None or _is_nan(v) else str(v))
            for k, v in auction_aliases.items()
        }
    else:
        auction_aliases = {}

    # Fuel aliases (normaliza keys)
    fuel_aliases = load_json(Path(args.fuel_aliases), {}) or {}
    if isinstance(fuel_aliases, dict):
        fuel_aliases = {normalize_text(str(k)): v for k, v in fuel_aliases.items()}
    else:
        fuel_aliases = {}

    tasas = load_tasas(tasas_path)

    # Lee Excel completo (NO filtramos sold)
    df = pd.read_excel(excel_path, sheet_name=args.sheet) if args.sheet else pd.read_excel(excel_path)

    # --------- Comisiones por fila ----------
    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        res = compute_commissions_for_row(row, tasas, country_dict, auction_aliases, fuel_aliases, include_not_sold, location_aliases)
        results.append(res)

    res_df = pd.DataFrame(results)

    # Inserta/actualiza columnas en df
    for c in res_df.columns:
        df[c] = res_df[c].values

    # Normaliza comisiones: NaN -> 0.0 y asegura tipo numérico
    comm_cols = [c for c in df.columns if c.startswith("commission_") and c.endswith("_eur")]
    for c in comm_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if "bca_commission_eur" in df.columns:
        df["bca_commission_eur"] = pd.to_numeric(df["bca_commission_eur"], errors="coerce").fillna(0.0)

    # Guarda Excel enriquecido
    if args.sheet and out_path == excel_path:
        with pd.ExcelWriter(out_path, mode="a", if_sheet_exists="replace", engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name=args.sheet, index=False)
    else:
        df.to_excel(out_path, index=False, engine="openpyxl")

    print(f"[OK] Comisiones calculadas (sin filtrar NOT SOLD) → {out_path}")


if __name__ == "__main__":
    main()