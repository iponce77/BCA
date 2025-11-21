# -*- coding: utf-8 -*-
import re
import argparse, sys, yaml
import pandas as pd
from pathlib import Path
import importlib.util

OUTPUT_COLS = [
    "link_ficha","make_clean","modelo_base_x","segmento","year","mileage","combustible_norm",
    "auction_name","winning_bid",
    "precio_final_eur","precio_venta_ganvam","margin_abs","vat_type",
    "units_abs_bcn","units_abs_cat","units_abs_esp","YoY_weighted_esp",
]



def normalize_fuel(x: str) -> str:
    if x is None: return ""
    v = str(x).strip().lower()
    if "bev" in v or "eléctr" in v or "electric" in v: return "BEV"
    if "diesel" in v or "diésel" in v: return "DIESEL"
    if "gas" in v and "lina" in v: return "GASOLINA"
    if "phev" in v or ("híbr" in v and "ench" in v) or "plug-in" in v: return "PHEV"
    if "hibr" in v or "hybrid" in v: return "HEV"
    return v.upper()

def filter_by_fuel(df, include=None, exclude=None):
    if "combustible_norm" not in df.columns and "fuel_type" not in df.columns:
        return df
    base = df.get("combustible_norm", df.get("fuel_type")).map(normalize_fuel)
    if include:
        inc = {normalize_fuel(x) for x in (include if isinstance(include,(list,set,tuple)) else [include])}
        df = df[base.isin(inc)]
    if exclude:
        exc = {normalize_fuel(x) for x in (exclude if isinstance(exclude,(list,set,tuple)) else [exclude])}
        df = df[~base.isin(exc)]
    return df

def normalize_transmission(x: str) -> str:
    if x is None: return ""
    v = str(x).strip().lower()
    if any(k in v for k in ["auto", "autom", "aut.", "automática", "automatica"]):
        return "automatic"
    if "manual" in v:
        return "manual"
    return v

def filter_by_transmission(df, want):
    if not want or "transmission" not in df.columns:
        return df
    if isinstance(want, (list, tuple, set)):
        target = set()
        for w in want:
            s = str(w).strip().lower()
            target.add("automatic" if "auto" in s else ("manual" if "man" in s else s))
        col = df["transmission"].map(normalize_transmission)
        return df[col.isin(target)]
    else:
        s = str(want).strip().lower()
        t = "automatic" if "auto" in s else ("manual" if "man" in s else s)
        col = df["transmission"].map(normalize_transmission)
        return df[col == t]

def load_module(path):
    spec = importlib.util.spec_from_file_location("bca_invest_recommender", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bca_invest_recommender"] = mod
    spec.loader.exec_module(mod)
    return mod

def normalize_lot_status(s: str) -> bool:
    if s is None:
        return False
    v = str(s).strip().lower()
    return (v == "sold") or ("sold" in v) or (v in {"vendido"}) or ("sale ended" in v) or ("closed" in v)

def sanitize_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    safe = re.sub(r'\s+', ' ', safe).strip()
    return safe

def apply_global_filters(df: pd.DataFrame, g: dict) -> pd.DataFrame:
    out = df.copy()

    # Transmission / Fuel (global)
    trans = g.get("transmission") or g.get("gearbox")
    if trans:
        out = filter_by_transmission(out, trans)
    fuel_inc = g.get("fuel_include") or g.get("combustible_include")
    fuel_exc = g.get("fuel_exclude") or g.get("combustible_exclude")
    out = filter_by_fuel(out, include=fuel_inc, exclude=fuel_exc)

    cap = g.get("margin_cap_ratio", 0.5)
    if "margin_abs" in out.columns and "precio_final_eur" in out.columns:
        out = out[out["margin_abs"] <= cap * out["precio_final_eur"]]

    if g.get("sold_only", True) and "lot_status" in out.columns:
        out = out[out["lot_status"].map(normalize_lot_status)]

    max_gap = g.get("max_year_gap", 3)
    if "year_bca" in out.columns and "anio_ganvam" in out.columns:
        out = out[(out["year_bca"] - out["anio_ganvam"]).abs() <= max_gap]

    return out

def ensure_output_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que todas las salidas tengan SOLO OUTPUT_COLS en ese orden.
       Si faltan campos, intenta rellenar mapeando/derivando con heurísticas
       consistentes con el motor."""
    out = df.copy()
    # Derivados/mapeos mínimos para layout
    if "auction_name" not in out.columns and "sale_name" in out.columns:
        out["auction_name"] = out["sale_name"]

    if "year" not in out.columns:
        for c in ["anio","Año","year_bca"]:
            if c in out.columns:
                out["year"] = out[c]
                break

    if "mileage" not in out.columns:
        for c in ["km","kilometros","kilómetros","odometro","odómetro"]:
            if c in out.columns:
                out["mileage"] = out[c]
                break

    # Asegurar combustible_norm como fuel normalizado
    if "combustible_norm" not in out.columns:
        if "fuel_type" in out.columns:
            # reutilizamos normalize_fuel definido arriba
            out["combustible_norm"] = out["fuel_type"].map(normalize_fuel)
        else:
            out["combustible_norm"] = pd.NA

    if "modelo_base_x" not in out.columns:
        for c in ["modelo_base", "modelo_base_y", "modelo_base_match", "modelo"]:
            if c in out.columns:
                out["modelo_base_x"] = out[c]
                break

    # OJO: eliminamos el bloque que creaba transmission-sale_country
    # (ya no forma parte de OUTPUT_COLS)

    for c in OUTPUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    return out[OUTPUT_COLS]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta al .parquet / .xlsx / .csv de BCA enriquecido")
    ap.add_argument("--yaml", required=True, help="Ruta al YAML de consultas")
    ap.add_argument("--outdir", default=".", help="Directorio de salida (CSV)")
    args = ap.parse_args()

    # Cargar módulo del motor
    mod = load_module(Path(__file__).parent / "bca_invest_recommender.py")
    df = mod.load_dataset(args.data)

    conf = yaml.safe_load(Path(args.yaml).read_text(encoding="utf-8"))
    alpha = float(conf.get("alpha", 0.6))
    global_filters = conf.get("global_filters", {})
    demand = conf.get("demand", {})

    # Filtro global de datos
    df = apply_global_filters(df, global_filters)

    # Build config
    DemandConfig = getattr(mod, "DemandConfig")
    RecommenderConfig = getattr(mod, "RecommenderConfig")
    dcfg = DemandConfig(
        use_brand_share=bool(demand.get("use_brand_share", True)),
        use_units_abs=bool(demand.get("use_units_abs", True)),
        use_concentration_penalty=bool(demand.get("use_concentration_penalty", False)),
        weight_brand_share=float(demand.get("weight_brand_share", 0.5)),
        weight_units_abs=float(demand.get("weight_units_abs", 0.5)),
        weight_concentration=float(demand.get("weight_concentration", 0.0)),
    )
    rcfg = RecommenderConfig(alpha_margin=alpha, demand=dcfg)

    rec = mod.BCAInvestRecommender(df, cfg=rcfg)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    results_info = []
    for q in conf.get("queries", []):
        qname = q.get("name", "query")
        qtype = q.get("type", "generic").lower()
        region = q.get("region", "bcn")
        filters = q.get("filters", {}) or {}
        prefer_fast = bool(q.get("prefer_fast", False))
        ignore_rotation = bool(q.get("ignore_rotation", False))
        brand_only = bool(q.get("brand_only", False))
        top_n = int(q.get("top_n", 10))
        selection = str(q.get("selection","cheapest")).lower()  # por defecto: "vehiculo óptimo"
        cluster = q.get("cluster", {}) or {}
        min_listings_per_group = int(q.get("min_listings_per_group", 1))
        prefer_cheapest_sort = bool(q.get("prefer_cheapest_sort", False))

        safe = sanitize_filename(qname)

        if qtype in {"best_auction_for_model","q1"}:
            model = q.get("model") or q.get("modelo") or ""
            top_listings, rank = rec.q1_best_auction_for_model(
                model_query=model, region=region, top_n=top_n,
                year_from=filters.get("year_from"), year_to=filters.get("year_to"),
                mileage_max=filters.get("mileage_max") or filters.get("max_km"),
            )
            f1 = outdir / f"{safe}.csv"
            ensure_output_cols(top_listings).to_csv(f1, index=False)
            results_info.append((qname, f1))

            f2 = outdir / f"{safe} - auction_ranking.csv"
            rank.to_csv(f2, index=False)
            results_info.append((qname + " (auction_ranking)", f2))
            continue

        if qtype in {"brand_price_order","q2"}:
            brand = q.get("brand") or q.get("marca") or ""
            filters_q = q.get("filters", {}) or {}

            # Parámetros estándar configurables
            min_year = filters_q.get("min_year", 2020)
            max_km = filters_q.get("max_km", 100000)

            # Modo de selección por modelo: "cheapest" o "max_margin"
            mode = q.get("mode", "cheapest")

            res = rec.q2_price_order_within_brand(
                brand=brand,
                region=region,
                min_year=min_year,
                max_km=max_km,
                mode=mode,
            )

            f = outdir / f"{safe}.csv"
            ensure_output_cols(res).to_csv(f, index=False)
            results_info.append((qname, f))
            continue

        if qtype in {"segment_price_order","q3"}:
            segment = q.get("segment") or q.get("segmento") or ""

            # lee filtros opcionales con alias habituales
            min_year = filters.get("min_year", filters.get("year_from", 2020))
            max_year = filters.get("max_year", filters.get("year_to",   2025))
            km_max   = filters.get("max_km",  filters.get("mileage_max", 100000))
            fuel     = filters.get("fuel",    filters.get("fuel_include"))

            res = rec.q3_price_order_within_segment(
                segment=segment,
                region=region,
                top_n=top_n,
                year_from=min_year,
                year_to=max_year,
                km_max=km_max,
                fuel_include=fuel,
            )
            f = outdir / f"{safe}.csv"
            ensure_output_cols(res).to_csv(f, index=False)
            results_info.append((qname, f))
            continue

        if qtype in {"best_fuel_gap","q5"}:
            model_base = q.get("modelo_base") or q.get("model_base") or q.get("modelo") or ""
            year = int(q.get("year") or q.get("anio") or q.get("año") or 0)
            res = rec.q5_best_fuel_gap(modelo_base=model_base, anio=year)
            f = outdir / f"{safe}.csv"
            res.to_csv(f, index=False)
            results_info.append((qname, f))
            continue

        # --- consultas genéricas (compat con V1) ---
        base_df = rec.df
        trans = filters.get("transmission") or filters.get("gearbox")
        if trans: base_df = filter_by_transmission(base_df, trans)
        fuel_inc = filters.get("fuel_include") or filters.get("combustible_include")
        fuel_exc = filters.get("fuel_exclude") or filters.get("combustible_exclude")
        base_df = filter_by_fuel(base_df, include=fuel_inc, exclude=fuel_exc)

        rec_local = mod.BCAInvestRecommender(base_df, cfg=rcfg)
        tmp = rec_local.recommend_best(
            region=region,
            max_age_years=filters.get("max_age_years"),
            max_price=filters.get("max_price"),
            min_price=filters.get("min_price"),
            ignore_rotation=ignore_rotation,
            prefer_fast=prefer_fast,
            brand_only=brand_only,
            selection=selection,  # default "cheapest"
            year_exact=filters.get("year_exact"),
            segment_include=filters.get("segment") or filters.get("segment_include"),
            segment_exclude=filters.get("segment_exclude"),
            mileage_min=filters.get("mileage_min"),
            mileage_max=filters.get("mileage_max") or filters.get("max_km"),
            include_sale_country=bool(cluster.get("include_sale_country", True)),
            include_sale_name=bool(cluster.get("include_sale_name", True)),
            min_listings_per_group=min_listings_per_group,
            prefer_cheapest_sort=prefer_cheapest_sort,
            n=top_n
        )
        f = outdir / f"{safe}.csv"
        ensure_output_cols(tmp).to_csv(f, index=False)
        results_info.append((qname, f))

    idx = pd.DataFrame([{"query": n, "file": str(p)} for n,p in results_info])
    idx.to_csv(outdir / "queries_index.csv", index=False)
    print(idx.to_string(index=False))

if __name__ == "__main__":
    main()
