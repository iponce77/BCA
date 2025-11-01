import polars as pl
import unicodedata
import re
from typing import Iterable, Optional

# -----------------------------
# Helpers
# -----------------------------
def _strip_accents_upper(s: Optional[str]) -> Optional[str]:
    if s is None:
        return s
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.upper()

def norm_upper(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
            .map_elements(_strip_accents_upper, return_dtype=pl.Utf8)
    )

# -----------------------------
# Column synonyms (minimal set)
# -----------------------------
COLUMN_SYNONYMS: dict[str, list[str]] = {
    "fecha_mat": ["fecha_mat","fecha","fecha_matriculacion","fecha matriculacion","fecha primera matriculacion","f_mat","fec_mat"],
    "marca": ["marca","make","fabricante","brand","make_clean"],
    "modelo": ["modelo","model","modelo_base","model_clean"],
    "tipo_vehiculo": ["tipo_vehiculo","tipo","segmento","categoria","tipo vehiculo"],
    "provincia": ["provincia","prov","provincia_dgt","province"],
    "codigo_provincia": ["codigo_provincia","cod_prov","ine_prov","ine"],
    "combustible": ["combustible","fuel","carburante","tipo_combustible"],
    "yyyymm": ["yyyymm","mes","periodo","period","yyyy_mm"],
    "anio": ["anio","ano","year"],
}

# Build a rename map based on present names (case/accents/underscore-insensitive)
def build_rename_map_from_names(names: Iterable[str]) -> dict[str, str]:
    present = list(names)
    norm_present = { _strip_accents_upper(n).replace(" ","").replace("-","").replace("_",""): n for n in present }
    rename: dict[str,str] = {}
    for canonical, syns in COLUMN_SYNONYMS.items():
        for s in [canonical] + syns:
            key = _strip_accents_upper(s).replace(" ","").replace("-","").replace("_","")
            if key in norm_present:
                rename[ norm_present[key] ] = canonical
                break
    return rename

# -----------------------------
# Public API expected by ETL:
#   - standardize_lazyframe()
#   - keep_tipos()
# -----------------------------
def _ensure_types_and_derivations(lf: pl.LazyFrame, yyyymm_hint: Optional[int]) -> pl.LazyFrame:
    cols = lf.columns

    # yyyymm from fecha_mat if needed
    if "yyyymm" not in cols:
        if "fecha_mat" in cols:
            lf = lf.with_columns([
                pl.col("fecha_mat").str.strptime(pl.Date, strict=False, fmt=None).dt.strftime("%Y%m").cast(pl.Int64).alias("yyyymm")
            ])
        elif yyyymm_hint is not None:
            lf = lf.with_columns(pl.lit(int(yyyymm_hint)).alias("yyyymm"))

    # anio from yyyymm
    if "anio" not in lf.columns and "yyyymm" in lf.columns:
        lf = lf.with_columns((pl.col("yyyymm") // 100).cast(pl.Int64).alias("anio"))

    # Normalized text versions
    if "marca" in lf.columns:
        lf = lf.with_columns(norm_upper(pl.col("marca")).alias("marca_normalizada"))
    if "modelo" in lf.columns:
        lf = lf.with_columns(norm_upper(pl.col("modelo")).alias("modelo_normalizado"))
    if "tipo_vehiculo" in lf.columns:
        lf = lf.with_columns(norm_upper(pl.col("tipo_vehiculo")).alias("tipo_vehiculo_norm"))
    if "combustible" in lf.columns:
        lf = lf.with_columns(norm_upper(pl.col("combustible")).alias("combustible_norm"))
    if "provincia" in lf.columns:
        lf = lf.with_columns(norm_upper(pl.col("provincia")).alias("provincia_norm"))

    return lf

def _apply_mappings(
    lf: pl.LazyFrame,
    brands_map: Optional[pl.DataFrame],
    fuels_map: Optional[pl.DataFrame],
) -> pl.LazyFrame:
    # brands_map expected columns: e.g. ["marca_normalizada","marca_final"]
    if brands_map is not None and "marca_normalizada" in lf.columns:
        b = brands_map
        join_key = "marca_normalizada" if "marca_normalizada" in b.columns else None
        if join_key:
            lf = lf.join(b.lazy(), on="marca_normalizada", how="left")
            # Prefer mapped column if present
            for cand in ["marca_final","marca_std","brand_std","brand"]:
                if cand in b.columns:
                    lf = lf.with_columns(
                        pl.coalesce([pl.col(cand), pl.col("marca_normalizada")]).alias("marca_normalizada")
                    )
                    break

    # fuels_map expected columns: e.g. ["combustible_norm","combustible_final"]
    if fuels_map is not None and "combustible_norm" in lf.columns:
        f = fuels_map
        join_key = "combustible_norm" if "combustible_norm" in f.columns else None
        if join_key:
            lf = lf.join(f.lazy(), on="combustible_norm", how="left")
            for cand in ["fuel_final","combustible_final","fuel_std"]:
                if cand in f.columns:
                    lf = lf.with_columns(
                        pl.coalesce([pl.col(cand), pl.col("combustible_norm")]).alias("combustible")
                    )
                    break
        # Ensure combustible exists
        if "combustible" not in lf.columns and "combustible_norm" in lf.columns:
            lf = lf.with_columns(pl.col("combustible_norm").alias("combustible"))

    return lf

def standardize_lazyframe(
    lf_raw: pl.LazyFrame,
    yyyymm_hint: Optional[int] = None,
    brands_map: Optional[pl.DataFrame] = None,
    fuels_map: Optional[pl.DataFrame] = None,
) -> pl.LazyFrame:
    """
    - Normaliza nombres de columnas usando COLUMN_SYNONYMS
    - Deriva yyyymm (desde fecha_mat o hint) y anio
    - Normaliza strings en mayúsculas sin acentos
    - Aplica mappings de marcas y combustibles (si se pasan)
    """
    # 1) rename columns to canonical
    rename = build_rename_map_from_names(lf_raw.columns)
    lf = lf_raw.rename(rename)

    # 2) ensure required basics
    lf = _ensure_types_and_derivations(lf, yyyymm_hint)

    # 3) apply mappings
    lf = _apply_mappings(lf, brands_map, fuels_map)

    # 4) enforce dtypes for key columns when present
    sel = []
    for c in ["yyyymm","anio","codigo_provincia","unidades"]:
        if c in lf.columns:
            sel.append(pl.col(c).cast(pl.Int64, strict=False).alias(c))
    if sel:
        lf = lf.with_columns(sel)

    return lf

def keep_tipos(lf: pl.LazyFrame, include: list[str]) -> pl.LazyFrame:
    if not include:
        return lf
    tokens = []
    for t in include:
        u = _strip_accents_upper(t or "").strip()
        u = u.replace("TODOTERRENO","TODO TERRENO").replace("TODOTERRENOS","TODO TERRENO")
        u = u.replace("TURISMOS","TURISMO")
        tokens.append(u)
    pat = "(" + "|".join([re.escape(x) for x in tokens]) + ")"
    col = "tipo_vehiculo_norm" if "tipo_vehiculo_norm" in lf.columns else "tipo_vehiculo"
    return lf.filter(pl.col(col).is_null() | pl.col(col).str.contains(pat))

