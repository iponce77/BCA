# bca_enrichment_pipeline.py
# -*- coding: utf-8 -*-
import argparse, os, json, unicodedata, math, random, datetime as dt
import pandas as pd
import numpy as np

# === Flexible IO helpers (added) ===
def _read_any(path):
    import pandas as pd
    p = str(path).lower()

    if p.endswith(".parquet"):
        return pd.read_parquet(path)

    if p.endswith(".xlsx") or p.endswith(".xls"):
        # Forzamos openpyxl para evitar autodetecciones raras en CI
        return pd.read_excel(path, engine="openpyxl")

    if p.endswith(".csv"):
        # CSV simple; si te topas con problemas de encoding, añade encoding="utf-8"
        return pd.read_csv(path)

    # --- fallbacks por si la extensión miente ---
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            return pd.read_csv(path)


def _to_any(df, out_path, sheet_name="bca_enriched"):
    import pandas as pd
    p = str(out_path).lower()
    if p.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    elif p.endswith(".csv"):
        df.to_csv(out_path, index=False)
    elif p.endswith(".xlsx") or p.endswith(".xls"):
        with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name=sheet_name, index=False)
    else:
        df.to_parquet(out_path, index=False)

# --- sanity helpers -------------------------------------------------
def sanity_report(bca_df, ine_df, merged_df, label="post-merge"):
    import pandas as pd  # por si no está en ámbito
    n_bca = len(bca_df) if isinstance(bca_df, pd.DataFrame) else -1
    n_ine = len(ine_df) if isinstance(ine_df, pd.DataFrame) else -1
    n_merged = len(merged_df) if isinstance(merged_df, pd.DataFrame) else -1
    ine_cols = [c for c in (merged_df.columns if n_merged >= 0 else [])
                if c in {"unidades","antiguedad_media","p25_antiguedad","p50_antiguedad","p75_antiguedad",
                         "mix_0_3_%","mix_4_7_%","mix_8mas_%"}]
    na_rate = (merged_df[ine_cols].isna().mean().mean() if ine_cols else 0.0) if n_merged >= 0 else 1.0
    print(f"[{label}] BCA={n_bca} INE={n_ine} MERGED={n_merged}  NA_rate(INE-cols)={na_rate:.2%}")

    if n_bca > 0 and n_merged == 0:
        print("!! ALERTA: merge produjo 0 filas (¿inner join, normalización, claves?)")
    if na_rate > 0.70:
        print("!! AVISO: >70% NA en columnas INE; posible problema de matching/normalización.")

def clean_concat(pool):
    import pandas as pd
    clean = []
    for df in pool:
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        # descarta DF sin filas, sin columnas o TODO-NA (todas las celdas NA)
        if df.empty or df.shape[1] == 0 or df.isna().all().all():
            continue
        clean.append(df)
    return pd.concat(clean, axis=0, ignore_index=True) if clean else pd.DataFrame()

def debug_pool(pool, label="pool"):
    import pandas as pd
    print(f"[{label}] size={len(pool)}")
    for i, df in enumerate(pool):
        if df is None:
            print(f"  [{i}] None")
        elif isinstance(df, pd.DataFrame):
            print(f"  [{i}] DataFrame filas={len(df)} empty={df.empty}")
        else:
            print(f"  [{i}] tipo={type(df).__name__}")


# =========================
# Utils y normalización
# =========================
def _normalize_str(x: str) -> str:
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    x = ''.join(c for c in unicodedata.normalize('NFKD', x) if not unicodedata.combining(c))
    x = ' '.join(x.split())
    return x if x != "" else None

def _safe_int(x):
    try: return int(x)
    except: return np.nan

def _to_yyyymm(x):
    if pd.isna(x): return None
    sx = str(x).strip()
    if len(sx)==6 and sx.isdigit(): return sx
    d = pd.to_datetime(x, errors="coerce")
    if pd.isna(d): return None
    return f"{d.year:04d}{d.month:02d}"

def _safe_div(num, den, return_nan=False):
    if den is None or pd.isna(den) or den == 0:
        return np.nan if return_nan else 0.0
    return num / den

def _dense_rank_desc(series: pd.Series) -> pd.Series:
    # Cambio quirúrgico: usar rank nativo, robusto con NaN (al fondo)
    return series.rank(method="dense", ascending=False, na_option="bottom")

# =========================
# Mapeos canónicos
# =========================
FUEL_MAP_BCA2INE = {
    # GASOLINA
    "GASOLINA":"GASOLINA","PETROL":"GASOLINA","HYBRID":"GASOLINA",
    "HEVPETROL":"GASOLINA","MHEVPETROL":"GASOLINA","PHEVPETROL":"GASOLINA",
    # DIESEL
    "DIESEL":"DIESEL","DIESEL.":"DIESEL","DIESEL/":"DIESEL","DIESEL-":"DIESEL",
    # BEV
    "ELECTRIC":"BEV","BEV":"BEV","ELECTRICO":"BEV","ELECTRICA":"BEV","EV":"BEV",
    # OTROS
    "CNG":"OTROS","LPG":"OTROS","BIFUEL":"OTROS","FLEXFUEL":"OTROS","HYDROGEN":"OTROS",
    "UNKNOWN":"OTROS","": "OTROS", None:"OTROS"
}

def map_fuel_bca_to_ine(df: pd.DataFrame, fuel_col: str) -> pd.Series:
    return df[fuel_col].map(_normalize_str).map(lambda f: FUEL_MAP_BCA2INE.get(f, "OTROS"))

def _map_region(v):
    vv = _normalize_str(v)
    if vv in (None,): return None
    if vv in ("BCN","BARCELONA"): return "BCN"
    if vv in ("CAT","CATALUNA","CATALUÑA","CATALUNYA"): return "CAT"
    if vv in ("ESP","ESPANA","ESPAÑA","NACIONAL","TOTAL"): return "ESP"
    if "BARCELONA" in vv: return "BCN"
    if "CATAL" in vv: return "CAT"
    if "ESP" in vv or "NACIO" in vv or "TOTAL" in vv: return "ESP"
    return vv

# =========================
# 1) Normalización BCA
# =========================

def normalize_brand_model(bca_df: pd.DataFrame,
                          make_col="make_clean",
                          model_col="modelo_base_y",
                          date_col="Fecha Matriculación",
                          fuel_col="fuel_type") -> pd.DataFrame:
    cols = {c:_normalize_str(c) for c in bca_df.columns}
    def _find(cands):
        for c in cands:
            key=_normalize_str(c)
            for orig,n in cols.items():
                if n==key: return orig
        return None
    make_col  = _find([make_col,"make","marca"])
    model_col = _find([model_col,"modelo_base","modelo"])
    date_col  = _find([date_col,"fecha_matriculacion","fecha"])
    fuel_col  = _find([fuel_col,"combustible"])

    if not all([make_col, model_col, date_col, fuel_col]):
        raise RuntimeError("BCA: faltan columnas mínimas (make/model/date/fuel).")

    # --- Robust parsing de fecha ---
    fecha_raw = bca_df[date_col]

    # 1) Forzar dayfirst para DD/MM/YYYY
    fecha_dt = pd.to_datetime(fecha_raw, errors="coerce", dayfirst=True)

    # 2) Invalidar años imposibles (ej. 0001)
    fecha_dt = fecha_dt.where(fecha_dt.dt.year > 1900)

    # 3) Si hay strings con solo año (ej. "2020"), recuperarlos
    mask_year_only = fecha_raw.astype(str).str.fullmatch(r"\d{4}", na=False)
    fecha_dt.loc[mask_year_only] = pd.to_datetime(
        fecha_raw[mask_year_only] + "-01-01", errors="coerce"
    )

    out = pd.DataFrame({
        "marca":  bca_df[make_col].map(_normalize_str),
        "modelo": bca_df[model_col].map(_normalize_str),
        "fecha_matriculacion": fecha_dt,
        "combustible_bca": bca_df[fuel_col].map(_normalize_str),
    })
    out["anio"] = out["fecha_matriculacion"].dt.year.astype("Int64")
    out["combustible_norm"] = map_fuel_bca_to_ine(out, "combustible_bca")
    return out

# =========================
# 2) Normalización INE/DGT
# =========================

def normalize_ine(ine_df: pd.DataFrame, muni_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza transmisiones INE y deriva región BCN/CAT/ESP usando municipios_ine.csv
    (columnas detectadas: codigo_ine, municipio, provincia).
    """
    # -------- Detectar columnas en agg_transmisiones_ine.csv --------
    def N(x):  # normaliza nombres para búsquedas flexibles
        return _normalize_str(x)

    ine_cols = {c: N(c) for c in ine_df.columns}
    def find_ine(*cands):
        for c in cands:
            k = N(c)
            for orig, n in ine_cols.items():
                if n == k:
                    return orig
        return None

    marca  = find_ine("marca_normalizada","marca","brand","marca_norm")
    modelo = find_ine("modelo_normalizado","modelo","model","modelo_norm")
    anio   = find_ine("anio","year","ano")
    fuel   = find_ine("combustible","fuel")
    units  = find_ine("unidades","units","volumen")
    codine = find_ine("codigo_ine","cod_ine","codine","codigo")
    yyyymm = find_ine("yyyymm","periodo","mes_id","period")

    if any(x is None for x in [marca, modelo, anio, fuel, units, codine]):
        raise RuntimeError(
            "INE/DGT: faltan columnas mínimas (marca/modelo/anio/combustible/unidades/codigo_ine)."
        )

    # -------- Base plana INE --------
    base = pd.DataFrame({
        "marca":        ine_df[marca].map(_normalize_str),
        "modelo":       ine_df[modelo].map(_normalize_str),
        "anio":         ine_df[anio].apply(_safe_int).astype("Int64"),
        "combustible":  ine_df[fuel].map(_normalize_str).map(lambda f: FUEL_MAP_BCA2INE.get(f, "OTROS")),
        "unidades":     pd.to_numeric(ine_df[units], errors="coerce").fillna(0.0).astype(float),
        "codigo_ine":   ine_df[codine]
    })
    base["yyyymm"] = ine_df[yyyymm].apply(_to_yyyymm) if yyyymm else None

    edad_media = find_ine("antiguedad_media","edad_media")
    p25 = find_ine("p25_antiguedad","p25_edad")
    p50 = find_ine("p50_antiguedad","p50_edad","mediana_antiguedad")
    p75 = find_ine("p75_antiguedad","p75_edad")
    mix03 = find_ine("%_0_3","mix_0_3_%","mix_0_3","mix_03_ratio")
    mix47 = find_ine("%_4_7","mix_4_7_%","mix_4_7","mix_47_ratio")
    mix8m = find_ine("%_8_mas","mix_8mas_%","mix_8_mas","mix_8plus_ratio")

    if edad_media: base["antiguedad_media"] = pd.to_numeric(ine_df[edad_media], errors="coerce")
    if p25:        base["p25_antiguedad"]   = pd.to_numeric(ine_df[p25], errors="coerce")
    if p50:        base["p50_antiguedad"]   = pd.to_numeric(ine_df[p50], errors="coerce")
    if p75:        base["p75_antiguedad"]   = pd.to_numeric(ine_df[p75], errors="coerce")
    if mix03:      base["mix_0_3_%"]        = pd.to_numeric(ine_df[mix03], errors="coerce")
    if mix47:      base["mix_4_7_%"]        = pd.to_numeric(ine_df[mix47], errors="coerce")
    if mix8m:      base["mix_8mas_%"]       = pd.to_numeric(ine_df[mix8m], errors="coerce")

    # -------- municipios_ine.csv: codigo_ine, municipio, provincia --------
    muni_cols = {c: N(c) for c in muni_df.columns}
    def find_muni(*cands):
        for c in cands:
            k = N(c)
            for orig, n in muni_cols.items():
                if n == k:
                    return orig
        return None

    m_cod = find_muni("codigo_ine","codigoine","cod_ine","codine")
    m_prov = find_muni("provincia","nombre_provincia","nomprov")

    if not m_cod or not m_prov:
        raise RuntimeError("municipios_ine.csv debe contener 'codigo_ine' y 'provincia'.")

    muni_small = muni_df[[m_cod, m_prov]].drop_duplicates().copy()
    muni_small.columns = ["codigo_ine", "provincia"]

    # ---------------------- [AÑADIDO AQUÍ] ----------------------
    # Harmonizar codigo_ine en ambos DFs (maneja 4 dígitos -> zfill(5), tipos int/str, etc.)
    def _ine_code_to_str(series: pd.Series) -> pd.Series:
        s = series.astype("string")
        s = s.str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
        s = s.str.zfill(5)  # si venía '1234' -> '01234'
        bad = s[~s.str.fullmatch(r"\d{5}", na=False)]
        if len(bad) > 0:
            raise ValueError(
                f"codigo_ine inválido en {len(bad)} filas (esperado 5 dígitos). Ejemplos: {bad.head(3).tolist()}"
            )
        return s

    base["codigo_ine"] = _ine_code_to_str(base["codigo_ine"])
    muni_small["codigo_ine"] = _ine_code_to_str(muni_small["codigo_ine"])
    # -------------------- [FIN AÑADIDO] -------------------------

    # Normalizamos provincia a texto canonical (upper/sin tildes)
    muni_small["provincia_norm"] = muni_small["provincia"].map(_normalize_str)

    # Merge por codigo_ine
    base = base.merge(muni_small[["codigo_ine","provincia_norm"]], on="codigo_ine", how="left")

    # -------- Derivar regiones --------
    # BCN: provincia == BARCELONA
    is_bcn = base["provincia_norm"].eq("BARCELONA")

    # CAT: provincias de Cataluña
    cat_set = {"BARCELONA","GIRONA","LLEIDA","TARRAGONA"}
    is_cat = base["provincia_norm"].isin(cat_set)

    bcn_df = base[is_bcn.fillna(False)].copy()
    bcn_df["region"] = "BCN"

    cat_df = base[is_cat.fillna(False)].copy()
    cat_df["region"] = "CAT"

    esp_df = base.copy()
    esp_df["region"] = "ESP"

    out = pd.concat([bcn_df, cat_df, esp_df], axis=0, ignore_index=True)
    return out[out["region"].isin(["BCN","CAT","ESP"])].copy()


# =========================
# 3) Agregados regionales
# =========================
def compute_stability_monthly(ine_norm: pd.DataFrame) -> pd.DataFrame:
    if ine_norm["yyyymm"].notna().any():
        monthly = (ine_norm.dropna(subset=["yyyymm"])
                   .groupby(["region","marca","modelo","anio","combustible","yyyymm"], as_index=False)
                   .agg(units_m=("unidades","sum")))
        stab = (monthly.groupby(["region","marca","modelo","anio","combustible"], as_index=False)
                .agg(stddev_mensual_units=("units_m","std"),
                     mean_mensual_units=("units_m","mean"),
                     meses_observados=("units_m","count")))
        stab["coef_var_%"] = np.where(stab["mean_mensual_units"].fillna(0)==0, np.nan,
                                      stab["stddev_mensual_units"]/stab["mean_mensual_units"])
    else:
        stab = ine_norm[["region","marca","modelo","anio","combustible"]].drop_duplicates().copy()
        stab["stddev_mensual_units"] = np.nan
        stab["coef_var_%"] = np.nan
    return stab[["region","marca","modelo","anio","combustible","stddev_mensual_units","coef_var_%"]]

def compute_trends(annual: pd.DataFrame) -> pd.DataFrame:
    df = annual.sort_values(["region","marca","modelo","combustible","anio"]).copy()
    df["units_prev"]  = df.groupby(["region","marca","modelo","combustible"])["units_abs"].shift(1)
    df["units_prev3"] = df.groupby(["region","marca","modelo","combustible"])["units_abs"].shift(3)
    df["YoY_%"]       = np.where((df["units_prev"].fillna(0)==0), np.nan, (df["units_abs"]-df["units_prev"])/df["units_prev"])
    df["Growth_3a_%"] = np.where((df["units_prev3"].fillna(0)==0), np.nan, (df["units_abs"]-df["units_prev3"])/df["units_prev3"])
    def _flag(y):
        if pd.isna(y): return np.nan
        if y > 0.10: return 1
        if y < -0.10: return 0
        return np.nan  # "otro si estable"
    df["trend_flag"] = df["YoY_%"].map(_flag)
    df["year_rank_in_model"] = df.groupby(["region","marca","modelo","combustible"])["units_abs"].transform(_dense_rank_desc)
    return df[["region","marca","modelo","anio","combustible","YoY_%","Growth_3a_%","trend_flag","year_rank_in_model"]]

def compute_shares_and_ranks(ine_norm: pd.DataFrame):
    # Annual units per cohort
    annual = (ine_norm.groupby(["region","marca","modelo","anio","combustible"], dropna=False, as_index=False)
              .agg(units_abs=("unidades","sum")))
    # Totals
    total_marca_year = (annual.groupby(["region","marca","anio"], as_index=False)
                        .agg(total_marca_year=("units_abs","sum")))
    total_combustible_year = (annual.groupby(["region","anio","combustible"], as_index=False)
                              .agg(total_combustible_year=("units_abs","sum")))
    total_year = (annual.groupby(["region","anio"], as_index=False)
                  .agg(total_year=("units_abs","sum")))
    # Model-level sums (all fuels) for ranks & brand dominance
    model_year_allfuels = (annual.groupby(["region","marca","modelo","anio"], as_index=False)
                           .agg(units_model_year=("units_abs","sum")))
    rank_general_base = model_year_allfuels.copy()
    rank_general_base["rank_general"] = rank_general_base.groupby(["region","anio"])["units_model_year"].transform(_dense_rank_desc)
    rank_brand_year_base = model_year_allfuels.copy()
    rank_brand_year_base["rank_brand_year"] = rank_brand_year_base.groupby(["region","marca","anio"])["units_model_year"].transform(_dense_rank_desc)
    rank_brand_year_base["is_top3_brand_year"] = (rank_brand_year_base["rank_brand_year"] <= 3).astype(int)
    rank_year_fuel_model_base = (annual.groupby(["region","marca","modelo","anio","combustible"], as_index=False)
                                 .agg(units_model_year_fuel=("units_abs","sum")))
    rank_year_fuel_model_base["rank_year_fuel_model"] = rank_year_fuel_model_base.groupby(["region","anio","combustible"])["units_model_year_fuel"].transform(_dense_rank_desc)
    # Best fuel (por modelo y año, en cada región)
    best_fuel_base = (annual.sort_values(["region","marca","modelo","anio","units_abs"], ascending=[True,True,True,True,False])
                      .groupby(["region","marca","modelo","anio"], as_index=False)
                      .first()[["region","marca","modelo","anio","combustible","units_abs"]]
                      .rename(columns={"combustible":"best_fuel","units_abs":"best_fuel_units"}))
    # Trends
    trends = compute_trends(annual.copy())
    # Monthly stability
    stability = compute_stability_monthly(ine_norm)
    # Concentration: brand dominance & HHI
    brand_year_model_shares = model_year_allfuels.merge(total_marca_year, on=["region","marca","anio"], how="left")
    brand_year_model_shares["share_model_in_brand"] = brand_year_model_shares.apply(
        lambda r: _safe_div(r["units_model_year"], r["total_marca_year"]), axis=1)
    hhi = (brand_year_model_shares.assign(share_sq=lambda df: df["share_model_in_brand"]**2)
           .groupby(["region","marca","anio"], as_index=False)
           .agg(HHI_marca=("share_sq","sum")))
    dominancia = brand_year_model_shares.rename(columns={"units_model_year": "units_modelo_en_marca_anio"})
    dominancia["dominancia_modelo_marca_%"] = dominancia.apply(
        lambda r: _safe_div(r["units_modelo_en_marca_anio"], r["total_marca_year"]), axis=1)

    # Demand structure (if present)
    edad_cols = [c for c in ["antiguedad_media","p50_antiguedad","p75_antiguedad","mix_0_3_%","mix_4_7_%","mix_8mas_%"]
                 if c in ine_norm.columns]
    if edad_cols:
        estructura = (ine_norm
                      .groupby(["region","marca","modelo","anio","combustible"], as_index=False)[edad_cols]
                      .mean(numeric_only=True))
    else:
        estructura = annual[["region","marca","modelo","anio","combustible"]].copy()
        for c in ["antiguedad_media","p50_antiguedad","p75_antiguedad","mix_0_3_%","mix_4_7_%","mix_8mas_%"]:
            estructura[c] = np.nan

    # ===== Base features por combustible =====
    features_fuel = (annual
        .merge(total_marca_year, on=["region","marca","anio"], how="left")
        .merge(total_combustible_year, on=["region","anio","combustible"], how="left")
        .merge(total_year, on=["region","anio"], how="left")
        .merge(rank_general_base[["region","marca","modelo","anio","rank_general"]],
               on=["region","marca","modelo","anio"], how="left")
        .merge(rank_brand_year_base[["region","marca","modelo","anio","rank_brand_year","is_top3_brand_year"]],
               on=["region","marca","modelo","anio"], how="left")
        .merge(rank_year_fuel_model_base[["region","marca","modelo","anio","combustible","rank_year_fuel_model"]],
               on=["region","marca","modelo","anio","combustible"], how="left")
        .merge(best_fuel_base, on=["region","marca","modelo","anio"], how="left")
        .merge(trends, on=["region","marca","modelo","anio","combustible"], how="left")
        .merge(stability, on=["region","marca","modelo","anio","combustible"], how="left")
        .merge(dominancia[["region","marca","modelo","anio","dominancia_modelo_marca_%"]],
               on=["region","marca","modelo","anio"], how="left")
        .merge(hhi, on=["region","marca","anio"], how="left")
        .merge(estructura, on=["region","marca","modelo","anio","combustible"], how="left")
    )
    features_fuel["share_marca_%"] = features_fuel.apply(lambda r: _safe_div(r["units_abs"], r["total_marca_year"]), axis=1)
    features_fuel["share_combustible_%"] = features_fuel.apply(lambda r: _safe_div(r["units_abs"], r["total_combustible_year"]), axis=1)
    features_fuel["share_año_%"] = features_fuel.apply(lambda r: _safe_div(r["units_abs"], r["total_year"]), axis=1)
    features_fuel["share_cohorte_%"] = features_fuel["share_combustible_%"]
    features_fuel["is_best_fuel"] = (features_fuel["combustible"]==features_fuel["best_fuel"]).astype("Int64")
    features_fuel["row_vs_best_fuel_%"] = features_fuel.apply(
        lambda r: np.nan if (pd.isna(r["best_fuel_units"]) or r["best_fuel_units"]==0) else r["units_abs"]/r["best_fuel_units"], axis=1)

    # ===== Agregado sin combustible =====
    features_nofuel = (
        features_fuel
        .groupby(["region","marca","modelo","anio"], as_index=False)
        .agg({
            "units_abs": "sum",
            "total_marca_year": "first",
            "total_year": "first",
            "rank_general": "first",
            "rank_brand_year": "first",
            "is_top3_brand_year": "first",
            "dominancia_modelo_marca_%": "first",
            "HHI_marca": "first",
            "antiguedad_media": "mean",
            "p50_antiguedad": "mean",
            "p75_antiguedad": "mean",
            "stddev_mensual_units": "mean",
            "coef_var_%": "mean",
            "best_fuel_units": "first",
            "best_fuel": "first",
        })
    )
    features_nofuel["share_marca_%"] = features_nofuel.apply(
        lambda r: _safe_div(r["units_abs"], r["total_marca_year"]), axis=1
    )
    features_nofuel["share_año_%"] = features_nofuel.apply(
        lambda r: _safe_div(r["units_abs"], r["total_year"]), axis=1
    )
    features_nofuel["share_combustible_%"] = np.nan
    features_nofuel["share_cohorte_%"] = np.nan
    features_nofuel["is_best_fuel"] = np.nan
    features_nofuel["row_vs_best_fuel_%"] = features_nofuel.apply(
        lambda r: np.nan
        if (pd.isna(r["best_fuel_units"]) or r["best_fuel_units"] == 0)
        else r["units_abs"] / r["best_fuel_units"],
        axis=1,
    )
    return features_fuel, features_nofuel, best_fuel_base

# =========================
# 4) Matching con fallbacks (corregido y estable)
# =========================
def _fuel_fallback_sequence(original: str):
    # Orden lógico propuesto: GASOLINA → DIESEL → BEV → OTROS (evita repeticiones)
    seq = ["GASOLINA", "DIESEL", "BEV", "OTROS"]
    o = original if original in seq else None
    # Ya se intentó el 'original' en el exacto; aquí devolvemos sólo alternativas
    return [f for f in seq if f != o]

def _prepare_candidate_exact(bca_key, base_fuel, geo):
    left = bca_key.copy()
    right = base_fuel[base_fuel["region"]==geo]
    m = left.merge(right, on=["marca","modelo","anio","combustible"], how="left", suffixes=("","_m"))
    m = m[m["units_abs"].notna()].copy()
    if not m.empty:
        m["match_kind"] = "exact"; m["geo_fallback"] = geo.lower()
    return m

def _prepare_candidate_fuel(bca_key, base_fuel, geo, fuel_value):
    left = bca_key.copy()
    left = left.assign(combustible=fuel_value)
    right = base_fuel[base_fuel["region"]==geo]
    m = left.merge(right, on=["marca","modelo","anio","combustible"], how="left", suffixes=("","_m"))
    m = m[m["units_abs"].notna()].copy()
    if not m.empty:
        m["match_kind"] = "fuel_fallback"; m["geo_fallback"] = geo.lower()
    return m

def _prepare_candidate_bestfuel(bca_key, base_fuel, best_fuel_base, geo):
    left = bca_key.copy()
    bf = best_fuel_base[best_fuel_base["region"]==geo][["marca","modelo","anio","best_fuel"]]
    left = left.merge(bf, on=["marca","modelo","anio"], how="left")
    left.loc[left["best_fuel"].notna(), "combustible"] = left.loc[left["best_fuel"].notna(), "best_fuel"]
    left = left.drop(columns=["best_fuel"])
    right = base_fuel[base_fuel["region"]==geo]
    m = left.merge(right, on=["marca","modelo","anio","combustible"], how="left")
    m = m[m["units_abs"].notna()].copy()
    if not m.empty:
        m["match_kind"] = "fuel_fallback"; m["geo_fallback"] = geo.lower()
    return m

def _prepare_candidate_nofuel(bca_key, base_nofuel, geo):
    left = bca_key.drop(columns=["combustible"]).copy()
    right = base_nofuel[base_nofuel["region"]==geo]
    m = left.merge(right, on=["marca","modelo","anio"], how="left", suffixes=("","_m"))
    m = m[m["units_abs"].notna()].copy()
    if not m.empty:
        m["match_kind"] = "no_fuel"; m["geo_fallback"] = geo.lower()
    return m

def _prioritized_union(bca_key, candidates, metric_cols):
    if not candidates:
        return pd.DataFrame(columns=["_bca_idx"] + metric_cols + ["match_kind","geo_fallback"])
    pool = []
    for cand in candidates:
        if cand is None or cand.empty:
            continue
        # Asegura presencia de clave
        if "_bca_idx" not in cand.columns:
            keys = [c for c in ["marca","modelo","anio","combustible"] if c in cand.columns]
            cand = cand.merge(bca_key[["_bca_idx","marca","modelo","anio","combustible"]], on=keys, how="left")
        # 🔒 Completar columnas métricas que falten
        for col in metric_cols + ["match_kind","geo_fallback"]:
            if col not in cand.columns:
                cand[col] = np.nan
        pool.append(cand[["_bca_idx"] + metric_cols + ["match_kind","geo_fallback"]])

    if not pool:
        return pd.DataFrame(columns=["_bca_idx"] + metric_cols + ["match_kind","geo_fallback"])
    merged = clean_concat(pool)
    merged = merged.drop_duplicates(subset=["_bca_idx"], keep="first")
    return merged

def _match_region(bca_key: pd.DataFrame, features_fuel, features_nofuel, best_fuel_base, region_code: str) -> pd.DataFrame:
    """
    Cambio quirúrgico: matching estable por _bca_idx y priorización:
    1) exacto → 2) fuel_fallback lógico → 3) best_fuel → 4) no_fuel
    """
    geo_order = {"BCN":["BCN","CAT","ESP"], "CAT":["CAT","ESP"], "ESP":["ESP"]}[region_code]

    # Métricas a traer (SIN 'rank_year_model')
    metrics = [
        "units_abs","share_marca_%","share_combustible_%","share_año_%","share_cohorte_%",
        "rank_general","rank_brand_year","rank_year_fuel_model","is_top3_brand_year",
        "best_fuel_units","best_fuel","is_best_fuel","row_vs_best_fuel_%",
        "YoY_%","Growth_3a_%","trend_flag","year_rank_in_model",
        "antiguedad_media","p50_antiguedad","p75_antiguedad","mix_0_3_%","mix_4_7_%","mix_8mas_%",
        "dominancia_modelo_marca_%","HHI_marca","stddev_mensual_units","coef_var_%"
    ]

    base = bca_key.reset_index(drop=False).rename(columns={"index":"_bca_idx"}).copy()

    all_candidates = []
    for geo in geo_order:
        # 1) EXÁCTO
        c_exact = _prepare_candidate_exact(base[["_bca_idx","marca","modelo","anio","combustible"]], features_fuel, geo)
        all_candidates.append(c_exact)

        # 2) FUEL FALLBACK LÓGICO
        # Para evitar intentar fuels duplicados, calculamos para cada fila la secuencia sin el combustible ya intentado.
        seq_map = {idx: _fuel_fallback_sequence(f) for idx, f in zip(base["_bca_idx"], base["combustible"])}
        # Iteramos en orden GASOLINA → DIESEL → BEV → OTROS
        for fallback_fuel in ["GASOLINA","DIESEL","BEV","OTROS"]:
            # Filas donde este combustible está en su secuencia
            idxs = [i for i, seq in seq_map.items() if fallback_fuel in seq]
            if not idxs:
                continue
            subset = base.loc[base["_bca_idx"].isin(idxs), ["_bca_idx","marca","modelo","anio"]].copy()
            subset["combustible"] = fallback_fuel
            c_fb = _prepare_candidate_fuel(subset, features_fuel, geo, fallback_fuel)
            all_candidates.append(c_fb)

        # 3) BEST_FUEL (solo para los que aún no hayan matcheado en este geo)
        c_best = _prepare_candidate_bestfuel(base[["_bca_idx","marca","modelo","anio","combustible"]], features_fuel, best_fuel_base, geo)
        all_candidates.append(c_best)

        # 4) NO_FUEL
        c_nofuel = _prepare_candidate_nofuel(base[["_bca_idx","marca","modelo","anio","combustible"]], features_nofuel, geo)
        all_candidates.append(c_nofuel)

    chosen = _prioritized_union(base, all_candidates, metrics)

    # Ensamblar salida alineando por _bca_idx, sin pre-crear columnas para evitar solapes
    out = base[["_bca_idx"]].merge(chosen, on="_bca_idx", how="left")

    # Asegurar que todas las columnas esperadas existan
    for c in metrics + ["match_kind","geo_fallback"]:
        if c not in out.columns:
            out[c] = np.nan

    # Ajuste: por definición actual, rank_year_model = rank_general
    out["rank_year_model"] = out["rank_general"]

    return out


def match_with_fallbacks(bca_norm: pd.DataFrame, ine_norm: pd.DataFrame) -> pd.DataFrame:
    features_fuel, features_nofuel, best_fuel_base = compute_shares_and_ranks(ine_norm)
    # clave BCA (con combustible normalizado como 'combustible')
    bca_key = bca_norm[["marca","modelo","anio","combustible_norm"]].rename(columns={"combustible_norm":"combustible"}).copy()
    # por región (cada uno devuelve _bca_idx + métricas + flags)
    mbcn = _match_region(bca_key, features_fuel, features_nofuel, best_fuel_base, "BCN")
    mcat = _match_region(bca_key, features_fuel, features_nofuel, best_fuel_base, "CAT")
    mesp = _match_region(bca_key, features_fuel, features_nofuel, best_fuel_base, "ESP")

    # sufijos
    def _sfx(df, suf):
        ren = {c:f"{c}_{suf}" for c in df.columns if c!="_bca_idx"}
        return df.rename(columns=ren)

    bcn_sfx = _sfx(mbcn,"bcn")
    cat_sfx = _sfx(mcat,"cat")
    esp_sfx = _sfx(mesp,"esp")

    enriched = (bca_norm.reset_index(drop=False).rename(columns={"index":"_bca_idx"})
                .merge(bcn_sfx, on="_bca_idx", how="left")
                .merge(cat_sfx, on="_bca_idx", how="left")
                .merge(esp_sfx, on="_bca_idx", how="left"))

    # comparativas
    enriched["ratio_bcn_cat_%"] = enriched.apply(lambda r: _safe_div(r.get("units_abs_bcn"), r.get("units_abs_cat"), return_nan=True), axis=1)
    enriched["ratio_cat_esp_%"] = enriched.apply(lambda r: _safe_div(r.get("units_abs_cat"), r.get("units_abs_esp"), return_nan=True), axis=1)
    return enriched

# =========================
# 5) Diccionario de datos (con notas)
# =========================
def build_data_dictionary() -> pd.DataFrame:
    rows=[]
    def add(name_base, desc, formula, per_region=True):
        if per_region:
            for suf in ["bcn","cat","esp"]:
                rows.append({
                    "nombre_columna": f"{name_base}_{suf}",
                    "descripcion": f"{desc} (región {suf.upper()})",
                    "formula": formula
                })
        else:
            rows.append({"nombre_columna": name_base, "descripcion": desc, "formula": formula})
    # Volumen y shares
    add("units_abs", "Volumen anual de transmisiones del cohorte (marca, modelo, año, combustible).",
        "Suma anual de 'unidades' en (marca, modelo, anio, combustible, región).")
    add("share_marca_%", "Cuota del modelo dentro de su marca en el año.", "units_abs / total_marca_(anio).")
    add("share_combustible_%", "Cuota del cohorte dentro del total del combustible en el año.",
        "units_abs / total_combustible_(anio).")
    add("share_año_%", "Cuota del cohorte dentro del total anual de la región.", "units_abs / total_año.")
    add("share_cohorte_%", "Cuota del cohorte (anio, combustible). (Nota: equivale a share_combustible_% bajo la definición actual).",
        "units_abs / total_(anio,combustible).")
    # Rankings
    add("rank_general", "Ranking general por año (todas marcas/modelos; suma combustibles).",
        "DENSE_RANK(desc) sobre units_model_year por (anio, región).")
    add("rank_brand_year", "Ranking dentro de la marca y año (suma combustibles).",
        "DENSE_RANK(desc) por (marca, anio, región).")
    add("rank_year_model", "Ranking por año a nivel modelo (suma combustibles). (Nota: actualmente equivalente a rank_general).",
        "DENSE_RANK(desc) por (anio, región).")
    add("rank_year_fuel_model", "Ranking por año y combustible a nivel modelo.",
        "DENSE_RANK(desc) por (anio, combustible, región).")
    add("is_top3_brand_year", "Indicador si el modelo está en el top 3 de su marca ese año.", "rank_brand_year ≤ 3.")
    # Combustible óptimo
    add("best_fuel_units","Unidades del combustible dominante del modelo en el año.",
        "Máximo units_abs por combustible en (marca, modelo, anio, región).")
    add("best_fuel","Combustible dominante del modelo en el año.",
        "Argmax combustible por units_abs en (marca, modelo, anio, región).")
    add("is_best_fuel","Indicador si la fila corresponde al combustible dominante.",
        "1 si combustible == best_fuel; 0 en otro caso; NaN si 'sin combustible'.")
    add("row_vs_best_fuel_%","Relación entre unidades de la fila y el combustible dominante.",
        "units_abs / best_fuel_units.")
    # Tendencias
    add("YoY_%","Crecimiento interanual del cohorte.","(units_t - units_{t-1}) / units_{t-1}; si units_{t-1}=0 → NaN.")
    add("Growth_3a_%","Crecimiento acumulado a 3 años del cohorte.","(units_t - units_{t-3}) / units_{t-3}; si units_{t-3}=0 → NaN.")
    add("trend_flag","Flag de tendencia","1 si YoY > +10%, 0 si YoY < −10%, NaN si estable. (No redefine meses faltantes).")
    add("year_rank_in_model","Ranking del año dentro del modelo+combustible.","DENSE_RANK(desc) por (marca,modelo,combustible,región).")
    # Estructura de demanda
    add("antiguedad_media","Antigüedad media de las transmisiones.","Promedio anual reportado por INE/DGT.")
    add("p50_antiguedad","Antigüedad mediana (p50).","Mediana anual reportada por INE/DGT.")
    add("p75_antiguedad","Antigüedad p75.","Percentil 75 anual reportado por INE/DGT.")
    add("mix_0_3_%","Mix 0–3 años.","Participación 0–3 años del cohorte.")
    add("mix_4_7_%","Mix 4–7 años.","Participación 4–7 años del cohorte.")
    add("mix_8mas_%","Mix 8+ años.","Participación 8+ años del cohorte.")
    # Concentración
    add("dominancia_modelo_marca_%","Participación del modelo dentro de su marca (año).",
        "units_modelo_en_marca_anio / total_marca_anio (suma combustibles).")
    add("HHI_marca","Índice de Herfindahl-Hirschman por marca y año (región).",
        "Suma de cuadrados de shares de cada modelo dentro de la marca (suma combustibles).")
    # Estabilidad
    add("stddev_mensual_units","Desviación estándar mensual de unidades del cohorte. (Nota: sólo meses observados, sin imputar ausentes).",
        "std de unidades mensuales (meses observados).")
    add("coef_var_%","Coeficiente de variación mensual del cohorte. (Nota: sólo meses observados).","std / mean (mensual).")
    # Comparativas (sin sufijo)
    add("ratio_bcn_cat_%","Relación de volumen BCN/CAT.","units_abs_bcn / units_abs_cat.", per_region=False)
    add("ratio_cat_esp_%","Relación de volumen CAT/ESP.","units_abs_cat / units_abs_esp.", per_region=False)
    return pd.DataFrame(rows)

# =========================
# 6) QA / Validaciones
# =========================
def build_qa(enriched: pd.DataFrame) -> dict:
    out = {}
    n = len(enriched)
    out["n_rows_bca"] = n
    out["versions"] = {"pandas": pd.__version__, "numpy": np.__version__, "run_timestamp": dt.datetime.utcnow().isoformat()+"Z"}
    for suf in ["bcn","cat","esp"]:
        mk = f"match_kind_{suf}"; gf=f"geo_fallback_{suf}"
        cov = enriched[mk].value_counts(dropna=False).to_dict()
        covg = enriched[gf].value_counts(dropna=False).to_dict()
        pct = lambda v: float(np.round((v/n)*100.0,2)) if n>0 else 0.0
        out[f"coverage_match_kind_{suf}"] = {k:pct(v) for k,v in cov.items()}
        out[f"coverage_geo_fallback_{suf}"] = {k:pct(v) for k,v in covg.items()}
    share_cols = [c for c in enriched.columns if c.startswith("share_") and c.endswith(("_bcn","_cat","_esp"))]
    out["share_range_violations"] = {c:int(((enriched[c].dropna()<0)|(enriched[c].dropna()>1)).sum()) for c in share_cols}
    rank_cols = [c for c in enriched.columns if c.startswith("rank_") and c.endswith(("_bcn","_cat","_esp"))]
    out["rank_min_values"] = {c:(float(enriched[c].dropna().min()) if enriched[c].notna().any() else np.nan) for c in rank_cols}
    bf_ok = {}
    for suf in ["bcn","cat","esp"]:
        cond = (enriched.get(f"is_best_fuel_{suf}")==1) & (enriched.get(f"best_fuel_units_{suf}").notna())
        v = (enriched.loc[cond, f"best_fuel_units_{suf}"] >= enriched.loc[cond, f"units_abs_{suf}"]).all()
        bf_ok[suf] = bool(v)
    out["best_fuel_consistency"] = bf_ok
    out["coverage_percent_bcn"] = float(np.round(100.0*enriched["match_kind_bcn"].notna().mean(),2))
    out["coverage_percent_esp"] = float(np.round(100.0*enriched["match_kind_esp"].notna().mean(),2))
    return out


# =========================
# 7) Runner end-to-end
# =========================
def run(bca_path: str, ine_path: str, muni_path: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    bca = _read_any(bca_path)
    ine = _read_any(ine_path)
    muni = _read_any(muni_path)

    bca_norm = normalize_brand_model(bca)
    ine_norm = normalize_ine(ine, muni)

    # 1) Calculamos solo las MÉTRICAS sobre la clave normalizada (con _bca_idx interno)
    enriched_metrics = match_with_fallbacks(bca_norm, ine_norm)

    # 2) Recuperamos TODAS las columnas originales de BCA y les añadimos _bca_idx
    bca_raw = bca.reset_index(drop=False).rename(columns={"index": "_bca_idx"})

    # 3) Evitamos duplicar columnas: si enriched_metrics trae alguna columna que ya existe en bca_raw,
    #    la retiramos antes del merge (salvo _bca_idx, que es la clave)
    cols_dup = [c for c in enriched_metrics.columns if c != "_bca_idx" and c in bca_raw.columns]
    enriched = bca_raw.merge(
        enriched_metrics.drop(columns=cols_dup, errors="ignore"),
        on="_bca_idx",
        how="left"
    )
    
    sanity_report(bca, ine_norm, enriched, label="enriched")
    sanity_report(bca, ine_norm, enriched, label="final-write")
    _to_any(enriched, os.path.join(outdir, "bca_enriched_final.parquet"))
    _to_any(enriched, os.path.join(outdir, "bca_enriched_final.xlsx"))
  
    # Diccionario
    data_dict = build_data_dictionary()
    data_dict.to_csv(os.path.join(outdir, "data_dictionary.csv"), index=False)
    with open(os.path.join(outdir, "data_dictionary.md"), "w", encoding="utf-8") as f:
        f.write("# Diccionario de datos\n\n")
        for _, r in data_dict.iterrows():
            f.write(f"**{r['nombre_columna']}** — {r['descripcion']}  \n*Fórmula:* {r['formula']}\n\n")

    # QA
    qa_stats = build_qa(enriched)
    pd.json_normalize(qa_stats, sep="__").T.reset_index().rename(columns={"index": "metric", 0: "value"}) \
        .to_csv(os.path.join(outdir, "qa_stats.csv"), index=False)
    with open(os.path.join(outdir, "qa_report.md"), "w", encoding="utf-8") as f:
        f.write("# Informe QA – Enriquecimiento BCA\n\n")
        f.write(json.dumps(qa_stats, ensure_ascii=False, indent=2))

    return {
        "out_parquet": os.path.join(outdir, "bca_enriched_final.parquet"),
        "out_xlsx": os.path.join(outdir, "bca_enriched_final.xlsx"),
        "data_dictionary_csv": os.path.join(outdir, "data_dictionary.csv"),
        "data_dictionary_md": os.path.join(outdir, "data_dictionary.md"),
        "qa_report_md": os.path.join(outdir, "qa_report.md"),
        "qa_stats_csv": os.path.join(outdir, "qa_stats.csv")
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bca", required=True, help="Ruta BCA (.parquet/.xlsx/.csv)")
    parser.add_argument("--ine", required=True, help="Ruta INE agg (.parquet/.csv/.xlsx)")
    parser.add_argument("--muni", required=True, help="Ruta a municipios_ine.csv")
    parser.add_argument("--outdir", default="./out", help="Directorio de salida")
    args = parser.parse_args()
    random.seed(42); np.random.seed(42)
    paths = run(args.bca, args.ine, args.muni, args.outdir)
    print(json.dumps(paths, indent=2, ensure_ascii=False))
