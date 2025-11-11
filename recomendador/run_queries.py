import re
import argparse, sys, yaml
import pandas as pd
from pathlib import Path
import importlib.util

def normalize_fuel(x: str) -> str:
    if x is None:
        return ""
    v = str(x).strip().lower()
    # normalización simple
    if "bev" in v or "eléctr" in v or "electric" in v:
        return "BEV"
    if "diesel" in v or "diésel" in v:
        return "DIESEL"
    if "gas" in v and "lina" in v:
        return "GASOLINA"
    if "phev" in v or ("híbr" in v and "ench" in v) or "plug-in" in v:
        return "PHEV"
    if "hibr" in v or "hybrid" in v:
        return "HEV"
    # deja tal cual si no reconocemos
    return v.upper()

def filter_by_fuel(df, include=None, exclude=None):
    if "combustible_norm" not in df.columns:
        return df
    col = df["combustible_norm"].map(normalize_fuel)
    if include:
        inc = {normalize_fuel(x) for x in (include if isinstance(include,(list,set,tuple)) else [include])}
        df = df[col.isin(inc)]
    if exclude:
        exc = {normalize_fuel(x) for x in (exclude if isinstance(exclude,(list,set,tuple)) else [exclude])}
        df = df[~col.isin(exc)]
    return df

def normalize_transmission(x: str) -> str:
    if x is None:
        return ""
    v = str(x).strip().lower()
    # patrones ES/EN típicos
    if any(k in v for k in ["auto", "autom", "aut.", "automática", "automatica"]):
        return "automatic"
    if any(k in v for k in ["manual"]):
        return "manual"
    return v

def filter_by_transmission(df, want):
    if not want or "transmission" not in df.columns:
        return df
    # permite string o lista
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
    # Sustituye caracteres prohibidos en Windows por "_"
    safe = re.sub(r'[<>:"/\\|?*]', '_', str(name))
    # Opcional: elimina dobles espacios y recorta
    safe = re.sub(r'\s+', ' ', safe).strip()
    return safe

def apply_global_filters(df: pd.DataFrame, g: dict) -> pd.DataFrame:
    out = df.copy()
    
    # Transmission (opcional)
    trans = g.get("transmission") or g.get("gearbox")
    if trans:
        out = filter_by_transmission(out, trans)

    # Fuel (opcional) — global
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ruta al .parquet / .xlsx / .csv de BCA enriquecido")
    ap.add_argument("--yaml", required=True, help="Ruta al YAML de consultas")
    ap.add_argument("--outdir", default=".", help="Directorio de salida (CSV)")
    args = ap.parse_args()

    mod = load_module(Path(__file__).parent / "bca_invest_recommender.py")
    df = mod.load_dataset(args.data)

    conf = yaml.safe_load(Path(args.yaml).read_text(encoding="utf-8"))
    alpha = float(conf.get("alpha", 0.6))
    global_filters = conf.get("global_filters", {})
    demand = conf.get("demand", {})

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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_info = []
    for q in conf.get("queries", []):
        qname = q.get("name", "query")
        qtype = q.get("type", "generic")
        region = q.get("region", "bcn")
        filters = q.get("filters", {}) or {}
        prefer_fast = bool(q.get("prefer_fast", False))
        ignore_rotation = bool(q.get("ignore_rotation", False))
        brand_only = bool(q.get("brand_only", False))
        top_n = int(q.get("top_n", 10))
        selection = str(q.get("selection","mean")).lower()
        cluster = q.get("cluster", {}) or {}
        min_listings_per_group = int(q.get("min_listings_per_group", 1))
        prefer_cheapest_sort = bool(q.get("prefer_cheapest_sort", False))

        # PATCH: consultas especiales paramétricas via YAML:
        #   special:
        #     model: <texto modelo>
        #     year: <int>
        special = q.get("special")
        if special:
            model = special.get("model")
            year = special.get("year")
            if model is None or year is None:
                raise ValueError(f"La consulta especial '{qname}' necesita 'model' y 'year' en 'special'.")
            res = rec.query_special(model, int(year))
            safe = sanitize_filename(qname)
            f1 = outdir / f"{safe} - by_country.csv"
            f2 = outdir / f"{safe} - by_subasta.csv"
            res["by_country"].to_csv(f1, index=False)
            res["by_subasta"].to_csv(f2, index=False)
            results_info.append((qname+" (by_country)", f1))
            results_info.append((qname+" (by_subasta)", f2))
            continue  # pasamos a la siguiente query
        else:
            # 2) Consultas normales (recomendaciones)

            # Filtro de transmisión por consulta (opcional)
            trans = filters.get("transmission") or filters.get("gearbox")
            base_df = rec.df
            if trans:
                base_df = filter_by_transmission(base_df, trans)

            # Filtro de combustible por consulta (opcional)
            fuel_inc = filters.get("fuel_include") or filters.get("combustible_include")
            fuel_exc = filters.get("fuel_exclude") or filters.get("combustible_exclude")
            base_df = filter_by_fuel(base_df, include=fuel_inc, exclude=fuel_exc)

            # Ya con todos los filtros aplicados, creamos el recomendador local
            rec_local = mod.BCAInvestRecommender(base_df, cfg=rcfg)

            tmp = rec_local.recommend_best(
                region=region,
                max_age_years=filters.get("max_age_years"),
                max_price=filters.get("max_price"),
                min_price=filters.get("min_price"),
                ignore_rotation=ignore_rotation,
                prefer_fast=prefer_fast,
                brand_only=brand_only,
                # NUEVO: selección y filtros avanzados
                selection=selection,  # "mean" | "cheapest"
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
            safe = sanitize_filename(qname)
            f = outdir / f"{safe}.csv"
            tmp.to_csv(f, index=False)
            results_info.append((qname, f))

    idx = pd.DataFrame([{"query": n, "file": str(p)} for n,p in results_info])
    idx.to_csv(outdir / "queries_index.csv", index=False)
    print(idx.to_string(index=False))

if __name__ == "__main__":
    main()
