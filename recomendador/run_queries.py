import argparse, sys, yaml
from pathlib import Path
import pandas as pd
import importlib.util
import re

# -------------------- Layout estándar --------------------
OUTPUT_COLS = [
    "link_ficha",
    "make_clean",
    "modelo_base_x",
    "segmento",
    "year_bca",
    "mileage",
    "fuel_type",
    "transmission",
    "sale_country",
    "sale_name",
    "winning_bid",
    "precio_final_eur",
    "precio_venta_ganvam",
    "margin_abs",
    "vat_type",
    "units_abs_bcn",
    "units_abs_cat",
    "units_abs_esp",
]

# -------------------- Normalizaciones ligeras --------------------
def normalize_fuel(x: str) -> str:
    if pd.isna(x):
        return ""
    v = str(x).strip().lower()
    if "bev" in v or "eléctr" in v or "electric" in v:
        return "BEV"
    if "diesel" in v or "diésel" in v:
        return "DIESEL"
    if ("gas" in v and "lina" in v) or v in {"gasolina","petrol"}:
        return "GASOLINA"
    if "híbr" in v or "hybrid" in v:
        return "HÍBRIDO"
    if "phev" in v:
        return "PHEV"
    if "gnc" in v or "cng" in v:
        return "GNC"
    if "gpl" in v or "lpg" in v:
        return "GPL"
    return v.upper()


def normalize_transmission(x: str) -> str:
    if pd.isna(x):
        return ""
    v = str(x).strip().lower()
    if "auto" in v:
        return "automatic"
    if "man" in v:
        return "manual"
    return v

# -------------------- ensure_output_cols --------------------
def ensure_output_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # link_ficha
    if "link_ficha" not in out.columns:
        for c in ["listing_url","lote_url","url","link"]:
            if c in out.columns:
                out["link_ficha"] = out[c]
                break
    if "link_ficha" not in out.columns:
        out["link_ficha"] = ""

    # make_clean (marca)
    if "make_clean" not in out.columns:
        out["make_clean"] = out["marca"] if "marca" in out.columns else ""

    # modelo_base_x con fallbacks
    if "modelo_base_x" not in out.columns:
        for c in ["modelo_base","modelo_base_y","modelo_base_match","modelo"]:
            if c in out.columns:
                out["modelo_base_x"] = out[c]
                break
    if "modelo_base_x" not in out.columns:
        out["modelo_base_x"] = ""

    # segmento
    if "segmento" not in out.columns:
        for c in ["segmento","segment","segmento_norm"]:
            if c in out.columns:
                out["segmento"] = out[c]
                break
    if "segmento" not in out.columns:
        out["segmento"] = ""

    # year_bca
    if "year_bca" not in out.columns:
        for c in ["anio","Año","year","year_bca"]:
            if c in out.columns:
                out["year_bca"] = pd.to_numeric(out[c], errors="coerce")
                break
    if "year_bca" not in out.columns:
        out["year_bca"] = pd.NA

    # mileage
    if "mileage" not in out.columns:
        for c in ["mileage","km","kilometros","kilómetros","odometro","odómetro"]:
            if c in out.columns:
                out["mileage"] = pd.to_numeric(out[c], errors="coerce")
                break
    if "mileage" not in out.columns:
        out["mileage"] = pd.NA

    # fuel_type
    if "fuel_type" not in out.columns:
        if "combustible_norm" in out.columns:
            out["fuel_type"] = out["combustible_norm"].map(normalize_fuel)
        else:
            out["fuel_type"] = ""

    # transmission (limpia)
    if "transmission" in out.columns:
        out["transmission"] = out["transmission"].map(normalize_transmission)
    else:
        out["transmission"] = ""

    # sale_country
    if "sale_country" not in out.columns:
        out["sale_country"] = ""

    # sale_name y mapeo auction_name->sale_name
    if "sale_name" not in out.columns:
        if "auction_name" in out.columns:
            out["sale_name"] = out["auction_name"]
        else:
            out["sale_name"] = ""

    # winning_bid (fallback a precio_final_eur)
    if "winning_bid" not in out.columns:
        out["winning_bid"] = out["precio_final_eur"] if "precio_final_eur" in out.columns else pd.NA

    # precios / margen
    if "precio_final_eur" not in out.columns:
        out["precio_final_eur"] = pd.NA
    if "precio_venta_ganvam" not in out.columns:
        out["precio_venta_ganvam"] = pd.NA
    if "margin_abs" not in out.columns:
        out["margin_abs"] = pd.NA

    # vat_type
    if "vat_type" not in out.columns:
        for c in ["vat_type","vat"]:
            if c in out.columns:
                out["vat_type"] = out[c]
                break
    if "vat_type" not in out.columns:
        out["vat_type"] = ""

    # units_abs_*
    for r in ("bcn","cat","esp"):
        col = f"units_abs_{r}"
        if col not in out.columns:
            out[col] = out[col] if col in out.columns else pd.NA

    # quitar columnas no deseadas si existen
    if "end_date" in out.columns:
        out = out.drop(columns=["end_date"])  

    # devolver SOLO OUTPUT_COLS, en orden
    for c in OUTPUT_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[OUTPUT_COLS]

# -------------------- util --------------------
def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[\\/:*?\"<>|]+", "_", str(name)).strip()
    return safe[:140] or "query"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("bca_invest_recommender", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bca_invest_recommender"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# -------------------- Global filters helpers --------------------
def apply_global_filters(df: pd.DataFrame, gf: dict) -> pd.DataFrame:
    out = df.copy()
    # margin_cap_ratio
    mcr = gf.get("margin_cap_ratio")
    if mcr is not None and "precio_final_eur" in out.columns and "margin_abs" in out.columns:
        price = pd.to_numeric(out["precio_final_eur"], errors="coerce")
        margin = pd.to_numeric(out["margin_abs"], errors="coerce")
        out = out[margin <= float(mcr) * price]
    # sold_only
    if gf.get("sold_only"):
        status_col = next((c for c in ["lot_status","status","estado"] if c in out.columns), None)
        if status_col:
            mask = out[status_col].astype(str).str.lower().str.contains(
                r"sold|ended|closed|vendido|sale ended"
            )
            out = out[mask]
    # max_year_gap (|year_bca - anio_ganvam| <= K)
    myg = gf.get("max_year_gap")
    if myg is not None and {"anio","anio_ganvam"}.issubset(out.columns):
        out = out[(out["anio"] - pd.to_numeric(out["anio_ganvam"], errors="coerce")).abs() <= int(myg)]
    # fuel_exclude
    fe = gf.get("fuel_exclude") or []
    if fe and "combustible_norm" in out.columns:
        fe_set = {str(x).strip().upper() for x in (fe if isinstance(fe,(list,tuple,set)) else [fe])}
        out = out[~out["combustible_norm"].astype(str).str.upper().isin(fe_set)]
    return out

# -------------------- runner --------------------
def run_from_yaml(dataset_path: Path, yaml_path: Path, outdir: Path):
    mod = load_module(Path(__file__).parent / "bca_invest_recommender.py")

    # cargar dataset con helper del módulo si existe
    if hasattr(mod, "load_dataset"):
        df = mod.load_dataset(str(dataset_path))
    else:
        # fallback sencillo
        if dataset_path.suffix.lower() in {".xlsx",".xls"}:
            df = pd.read_excel(dataset_path)
        elif dataset_path.suffix.lower() in {".csv",".txt"}:
            df = pd.read_csv(dataset_path)
        elif dataset_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(dataset_path)
        else:
            raise ValueError("Formato no soportado. Usa .parquet, .xlsx o .csv")

    conf = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))

    # aplicar global filters si existen
    gf = conf.get("global_filters", {}) or {}
    if gf:
        df = apply_global_filters(df, gf)

    # inicializar recomendador con alpha / demand del YAML si aparecen
    alpha = float(conf.get("alpha", 0.6))
    dem = conf.get("demand", {}) or {}
    demand_cfg = mod.DemandConfig(
        use_brand_share=bool(dem.get("use_brand_share", True)),
        use_units_abs=bool(dem.get("use_units_abs", True)),
        use_concentration_penalty=bool(dem.get("use_concentration_penalty", False)),
        weight_brand_share=float(dem.get("weight_brand_share", 0.5)),
        weight_units_abs=float(dem.get("weight_units_abs", 0.5)),
        weight_concentration=float(dem.get("weight_concentration", 0.25)),
    )
    cfg = mod.RecommenderConfig(alpha_margin=alpha, demand=demand_cfg)

    rec = mod.BCAInvestRecommender(df, cfg)

    outdir.mkdir(parents=True, exist_ok=True)

    results_info = []
    for q in conf.get("queries", []):
        qname   = q.get("name", "query")
        qtype   = str(q.get("type", "generic")).lower()
        region  = q.get("region", "bcn")
        filters = q.get("filters", {}) or {}
        prefer_fast   = bool(q.get("prefer_fast", False))
        ignore_rot    = bool(q.get("ignore_rotation", False))
        brand_only    = bool(q.get("brand_only", False))
        top_n         = int(q.get("top_n", 10))
        selection     = str(q.get("selection", "mean")).lower()  # "cheapest"|"mean"
        min_group     = int(q.get("min_listings_per_group", 1))
        prefer_cheapest_sort = bool(q.get("prefer_cheapest_sort", False))

        # ----- Tipos especiales (comparativas por modelo/año) -----
        special = q.get("special")
        if special:
            model = special.get("model"); year = special.get("year")
            if model is None or year is None:
                raise ValueError(f"La consulta especial '{qname}' necesita 'model' y 'year'.")
            res = rec.query_special(model, int(year))
            safe = sanitize_filename(qname)
            f1 = outdir / f"{safe} - by_country.csv"
            f2 = outdir / f"{safe} - by_subasta.csv"
            ensure_output_cols(res["by_country"]).to_csv(f1, index=False)
            ensure_output_cols(res["by_subasta"]).to_csv(f2, index=False)
            results_info.append((qname+" (by_country)", f1))
            results_info.append((qname+" (by_subasta)", f2))
            continue

        # ----- Q3: orden de precio por segmento (cheapest por cluster -> Top-N por margen) -----
        if qtype in {"q3","segment_price_order"}:
            segment = q.get("segment") or q.get("segmento")
            if not segment:
                raise ValueError(f"La consulta '{qname}' necesita 'segment' (por ejemplo: 'B' o 'SUV COMPACTO')")
            min_year = int(filters.get("min_year", 2020))
            max_year = int(filters.get("max_year", 2025))
            max_km   = filters.get("max_km", 100000)
            fuel     = filters.get("fuel")  # string o lista opcional

            tmp = rec.q3_segment_price_order(
                region=region,
                segment=segment,
                n=top_n,
                min_year=min_year,
                max_year=max_year,
                mileage_max=max_km,
                fuel_include=fuel,
            )

            tmp = ensure_output_cols(tmp)
            safe = sanitize_filename(qname)
            f = outdir / f"{safe}.csv"
            tmp.to_csv(f, index=False)
            results_info.append((qname, f))
            continue

        # ----- Q1/Q2/GENÉRICAS: recommend_best -----
        year_exact = filters.get("year_exact")
        min_year   = filters.get("min_year")
        max_year   = filters.get("max_year")
        max_km     = filters.get("max_km") or filters.get("mileage_max")

        # filtros previos directos (para acelerar)
        if min_year is not None and "anio" in rec.df.columns:
            rec.df = rec.df[rec.df["anio"] >= int(min_year)].copy()
        if max_year is not None and "anio" in rec.df.columns:
            rec.df = rec.df[rec.df["anio"] <= int(max_year)].copy()
        if max_km is not None:
            km_col = next((c for c in ["km","kilometros","kilómetros","mileage","odometro","odómetro"] if c in rec.df.columns), None)
            if km_col:
                rec.df = rec.df[pd.to_numeric(rec.df[km_col], errors="coerce") <= float(max_km)].copy()

        tmp = rec.recommend_best(
            region=region,
            ignore_rotation=ignore_rot,
            prefer_fast=prefer_fast,
            brand_only=brand_only,
            selection=selection,
            year_exact=year_exact,
            min_year=min_year,
            max_year=max_year,
            mileage_max=max_km,
            min_listings_per_group=min_group,
            prefer_cheapest_sort=prefer_cheapest_sort,
            n=top_n,
        )

        # Siempre formatear al layout estándar
        tmp = ensure_output_cols(tmp)
        safe = sanitize_filename(qname)
        f = outdir / f"{safe}.csv"
        tmp.to_csv(f, index=False)
        results_info.append((qname, f))

    # Índice de salidas
    idx = pd.DataFrame([{"query": n, "file": str(p)} for n,p in results_info])
    idx.to_csv(outdir / "queries_index.csv", index=False)
    print(idx.to_string(index=False))

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Runner de consultas BCA (Q1/Q2/Q3 + layout estándar)")
    ap.add_argument("--dataset", required=True, type=Path, help="Ruta del dataset (.parquet/.xlsx/.csv)")
    ap.add_argument("--yaml", required=True, type=Path, help="YAML con la lista de queries")
    ap.add_argument("--outdir", required=True, type=Path, help="Directorio de salida")
    args = ap.parse_args()
    run_from_yaml(args.dataset, args.yaml, args.outdir)

if __name__ == "__main__":
    main()
