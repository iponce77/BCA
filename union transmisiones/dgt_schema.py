# BCA/union transmisiones/dgt_schema.py
import re
import unicodedata
from typing import Iterable, Optional

import polars as pl


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
    "fecha_mat": [
        "fecha_mat",
        "fecha",
        "fecha_matriculacion",
        "fecha matriculacion",
        "fecha primera matriculacion",
        "f_mat",
        "fec_mat",
    ],
    "marca": ["marca", "make", "fabricante", "brand", "make_clean"],
    "modelo": ["modelo", "model", "modelo_base", "model_clean"],
    "tipo_vehiculo": ["tipo_vehiculo", "tipo", "segmento", "categoria", "tipo vehiculo"],
    "provincia": ["provincia", "prov", "provincia_dgt", "province"],
    "codigo_provincia": ["codigo_provincia", "cod_prov", "ine_prov", "ine"],
    "combustible": ["combustible", "fuel", "carburante", "tipo_combustible"],
    "yyyymm": ["yyyymm", "mes", "periodo", "period", "yyyy_mm"],
    "anio": ["anio", "ano", "year"],
}


def build_rename_map_from_names(names: Iterable[str]) -> dict[str, str]:
    """
    Dado el listado real de columnas (names), construye un mapa de renombre
    hacia los nombres canónicos de COLUMN_SYNONYMS.
    """
    norm_present = {
        _strip_accents_upper(n).replace(" ", "").replace("-", "").replace("_", ""): n
        for n in names
    }
    rename: dict[str, str] = {}
    for canonical, syns in COLUMN_SYNONYMS.items():
        for s in [canonical] + syns:
            key = _strip_accents_upper(s).replace(" ", "").replace("-", "").replace("_", "")
            if key in norm_present:
                rename[norm_present[key]] = canonical
                break
    return rename


# -----------------------------
# Public API expected by ETL:
#   - standardize_lazyframe()
#   - keep_tipos()
# -----------------------------
def _ensure_types_and_derivations(lf: pl.LazyFrame, yyyymm_hint: Optional[int]) -> pl.LazyFrame:
    """
    - Deriva yyyymm a partir de fecha_mat cuando no exista (o usa yyyymm_hint).
    - Deriva anio desde yyyymm si falta.
    - Crea versiones normalizadas en mayúsculas sin acentos para varias columnas.
    Evita accesos costosos a `lf.columns` usando `collect_schema().names()`.
    """
    names = lf.collect_schema().names()

    # yyyymm from fecha_mat if needed
    if "yyyymm" not in names:
        if "fecha_mat" in names:
            lf = lf.with_columns(
                pl.col("fecha_mat")
                .str.to_date(format=None, strict=False)
                .dt.strftime("%Y%m")
                .cast(pl.Int64)
                .alias("yyyymm")
            )
        elif yyyymm_hint is not None:
            lf = lf.with_columns(pl.lit(int(yyyymm_hint)).alias("yyyymm"))

    names = lf.collect_schema().names()

    # anio from yyyymm
    if "anio" not in names and "yyyymm" in names:
        lf = lf.with_columns((pl.col("yyyymm") // 100).cast(pl.Int64).alias("anio"))

    names = lf.collect_schema().names()

    # Normalized text versions (solo si existen)
    add_cols: list[pl.Expr] = []
    if "marca" in names:
        add_cols.append(norm_upper(pl.col("marca")).alias("marca_normalizada"))
    if "modelo" in names:
        add_cols.append(norm_upper(pl.col("modelo")).alias("modelo_normalizado"))
    if "tipo_vehiculo" in names:
        add_cols.append(norm_upper(pl.col("tipo_vehiculo")).alias("tipo_vehiculo_norm"))
    if "combustible" in names:
        add_cols.append(norm_upper(pl.col("combustible")).alias("combustible_norm"))
    if "provincia" in names:
        add_cols.append(norm_upper(pl.col("provincia")).alias("provincia_norm"))
    if add_cols:
        lf = lf.with_columns(add_cols)

    return lf


def _apply_mappings(
    lf: pl.LazyFrame,
    brands_map: Optional[pl.DataFrame],
    fuels_map: Optional[pl.DataFrame],
) -> pl.LazyFrame:
    """
    - Aplica mappings de marcas y combustibles si se proporcionan.
    - No asume nombres exactos en mapas; intenta candidatos razonables.
    """
    # brands_map expected columns: e.g. ["marca_normalizada","marca_final"]
    if brands_map is not None:
        b = brands_map
        b_names = set(b.columns)
        if "marca_normalizada" in lf.collect_schema().names() and (
            "marca_normalizada" in b_names or len(b_names) > 1
        ):
            lf = lf.join(b.lazy(), on="marca_normalizada", how="left")
            # Prefer mapped column if present
            for cand in ("marca_final", "marca_std", "brand_std", "brand"):
                if cand in b_names:
                    lf = lf.with_columns(
                        pl.coalesce([pl.col(cand), pl.col("marca_normalizada")]).alias("marca_normalizada")
                    )
                    break

    # fuels_map expected columns: e.g. ["combustible_norm","combustible_final"]
    if fuels_map is not None:
        f = fuels_map
        f_names = set(f.columns)
        lf_names = set(lf.collect_schema().names())

        if "combustible_norm" in lf_names and ("combustible_norm" in f_names or len(f_names) > 1):
            lf = lf.join(f.lazy(), on="combustible_norm", how="left")
            for cand in ("fuel_final", "combustible_final", "fuel_std"):
                if cand in f_names:
                    lf = lf.with_columns(
                        pl.coalesce([pl.col(cand), pl.col("combustible_norm")]).alias("combustible")
                    )
                    break

        # Ensure combustible existe aunque no haya mapeo explícito
        lf_names = set(lf.collect_schema().names())
        if "combustible" not in lf_names and "combustible_norm" in lf_names:
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
    - Normaliza strings (MAYÚSCULAS sin acentos)
    - Aplica mappings de marcas y combustibles (si se pasan)
    - Fuerza dtypes básicos
    """
    # 1) rename columns to canonical
    names0 = lf_raw.collect_schema().names()
    rename = build_rename_map_from_names(names0)
    lf = lf_raw.rename(rename)

    # 2) ensure required basics
    lf = _ensure_types_and_derivations(lf, yyyymm_hint)

    # 3) apply mappings
    lf = _apply_mappings(lf, brands_map, fuels_map)

    # 4) enforce dtypes para columnas clave cuando existan
    namesX = lf.collect_schema().names()
    cast_exprs: list[pl.Expr] = []
    for c in ("yyyymm", "anio", "codigo_provincia", "unidades"):
        if c in namesX:
            cast_exprs.append(pl.col(c).cast(pl.Int64, strict=False).alias(c))
    if cast_exprs:
        lf = lf.with_columns(cast_exprs)

    return lf


def keep_tipos(lf: pl.LazyFrame, include: list[str]) -> pl.LazyFrame:
    """
    Filtra el LazyFrame para mantener filas cuyo tipo_vehiculo contenga alguno de los tokens indicados.
    Usa la columna normalizada si existe; si no, la original.
    """
    if not include:
        return lf

    tokens: list[str] = []
    for t in include:
        u = _strip_accents_upper((t or "").strip())
        # Normalizaciones suaves
        u = u.replace("TODOTERRENO", "TODO TERRENO").replace("TODOTERRENOS", "TODO TERRENO")
        u = u.replace("TURISMOS", "TURISMO")
        tokens.append(u)

    pat = "(" + "|".join([re.escape(x) for x in tokens]) + ")"
    names = lf.collect_schema().names()
    col = "tipo_vehiculo_norm" if "tipo_vehiculo_norm" in names else (
        "tipo_vehiculo" if "tipo_vehiculo" in names else None
    )
    if col is None:
        # Si no hay columna, no filtra (devuelve tal cual)
        return lf

    return lf.filter(pl.col(col).is_null() | pl.col(col).str.contains(pat))
