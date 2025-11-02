from __future__ import annotations
import polars as pl
import unicodedata
import re

# ------------------------------------------------------------
# Normalización básica
# ------------------------------------------------------------
def _strip_accents_upper(s: str) -> str:
    if s is None:
        return s
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.upper().strip()

def norm_upper(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8, strict=False)
            .map_elements(_strip_accents_upper, return_dtype=pl.Utf8)
    )

# ------------------------------------------------------------
# Sinónimos de columnas según el esquema DGT real aportado
# ------------------------------------------------------------
COLUMN_SYNONYMS = {
    "fecha_mat": ["fecha_mat","fecha","fecha_matriculacion","fecha matriculación","fecha primera matriculacion","f_mat"],
    "marca": ["marca","make","fabricante","brand","make_clean"],
    "modelo": ["modelo","model","modelo_bare"],
    "submodelo": ["submodelo","version","versión"],
    "vin": ["vin","bastidor","num_bastidor"],
    "combustible": ["combustible","fuel","tipo_combustible"],
    "codigo_ine": ["codigo_ine","cod_ine"],
    "localidad": ["localidad","municipio","poblacion","población"],
    "provincia": ["provincia"],
    "tipo_vehiculo": ["tipo_vehiculo","tipo","categoria","categoría"],
    # OJO: 'transmision' es un campo de evento/lote (puede contener YYYY-MM / nombres), NO caja de cambios
    "transmision": ["transmision","transmisiones","lote","evento"],
    "año_mat": ["año_mat","ano_mat","aÃ±o_mat","anio_mat"],
    "mes_mat": ["mes_mat","mes_matriculacion","mes"],
    "antiguedad_anios": ["antiguedad_anios","antigüedad","antiguedad","edad_anios","edad_anos"],
    "nombre_archivo": ["nombre_archivo","source_file","file"],
    "es_cruzable": ["es_cruzable","cruzable"],
    "codigo_provincia": ["codigo_provincia","cod_prov","ineprov","prov"],
    "modelo_base": ["modelo_base","modelo canonico","modelo_canonico"],
    "make_clean": ["make_clean","marca_limpia"],
    # Clave final del join
    "modelo_normalizado": ["modelo_normalizado","modelo_norm","model_normalized"],
}

def build_rename_map_from_names(names: list[str]) -> dict[str, str]:
    lo = [c.lower() for c in names]
    rename = {}
    for canon, syns in COLUMN_SYNONYMS.items():
        for s in syns:
            s_low = s.lower()
            if s_low in lo:
                i = lo.index(s_low)
                old = names[i]
                if old != canon:
                    rename[old] = canon
                break
    return rename

def safe_rename_first(lf: pl.LazyFrame) -> pl.LazyFrame:
    # 1) Limpia nombres crudos con espacios o saltos de línea
    names0 = lf.collect_schema().names()
    cleaned = {old: old.strip().replace("\r", "").replace("\n", "")
               for old in names0
               if old != old.strip() or ("\r" in old) or ("\n" in old)}
    if cleaned:
        lf = lf.rename(cleaned)

    # 2) Aplica sinónimos → canónicos
    names = lf.collect_schema().names()
    rm = build_rename_map_from_names(names)

    # 3) Evita colisiones: si ya existe el canónico, no renombra ese sinónimo
    names_set = set(names)
    rm_safe = {old: canon for old, canon in rm.items() if canon not in names_set}
    if rm_safe:
        lf = lf.rename(rm_safe)
    return lf

# ------------------------------------------------------------
# Helpers de extracción
# ------------------------------------------------------------
def _yyyymm_from_text_expr(text_expr: pl.Expr) -> pl.Expr:
    """
    Extrae YYYYMM robusto a partir de un texto:
      - si hay 'YYYY-MM' o 'YYYY/MM' o '...YYYYMM...' lo devuelve como Int64
    """
    cleaned = text_expr.cast(pl.Utf8, strict=False).str.replace_all(r"[-/]", "", literal=False)
    six = cleaned.str.extract(r"(\d{6})", group_index=1)
    return six.cast(pl.Int64, strict=False)

def yyyymm_from_sources(yyyymm_hint: int | None) -> pl.Expr:
    trans = _yyyymm_from_text_expr(pl.col("transmision"))
    fname = _yyyymm_from_text_expr(pl.col("nombre_archivo"))
    hint = pl.lit(yyyymm_hint, dtype=pl.Int64)
    return pl.coalesce([trans, fname, hint]).alias("yyyymm")

def year_from_yyyymm(expr: pl.Expr) -> pl.Expr:
    return pl.floor(expr.cast(pl.Float64) / 100.0).cast(pl.Int64)

# ------------------------------------------------------------
# API principal para el ETL
# ------------------------------------------------------------
def standardize_lazyframe(
    lf: pl.LazyFrame,
    yyyymm_hint: int | None,
    brands_map: pl.DataFrame | None,
    fuels_map: pl.DataFrame | None,
) -> pl.LazyFrame:
    """
    Estandariza columnas y claves del DGT conforme al esquema real y a las reglas pedidas.
    - yyyymm: evento/lote → transmision (YYYY-MM / YYYY/MM / contiene YYYYMM) → nombre_archivo → hint
    - anio: año de primera matriculación → año_mat → fecha_mat → derivación por antigüedad vs yyyymm → año(yyyymm)
    - Normalizaciones: marca_normalizada = coalesce(make_clean, marca); modelo_normalizado = coalesce(modelo_base, modelo)
      Todo en UPPER y strip. Joins/keys en UPPER (código provincia como string).
    """
    lf = safe_rename_first(lf)

    # Parseo de fecha de matriculación (formatos mixtos)
    lf = lf.with_columns(
        pl.col("fecha_mat").cast(pl.Utf8).str.strptime(pl.Date, "%d/%m/%Y", strict=False)
          .fill_null(pl.col("fecha_mat").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False))
          .alias("fecha_mat_parsed")
    )

    # yyyymm a partir de transmision / nombre_archivo / hint
    lf = lf.with_columns(yyyymm_from_sources(yyyymm_hint))

    # Marca y Modelo canonizados (coalesce + UPPER) con mapeos opcionales
    marca_src = pl.coalesce([pl.col("make_clean"), pl.col("marca")]).alias("_marca_src")
    modelo_src = pl.coalesce([pl.col("modelo_base"), pl.col("modelo")]).alias("_modelo_src")
    lf = lf.with_columns([norm_upper(marca_src).alias("_marca_alias_norm"),
                          norm_upper(modelo_src).alias("_modelo_norm")])

    if brands_map is not None and brands_map.height > 0:
        # join en alias_norm → marca_normalizada (canónica) o alias si no hay mapping
        lf = (lf.join(brands_map.lazy(), left_on="_marca_alias_norm", right_on="alias_norm", how="left")
                .with_columns(pl.coalesce([pl.col("marca_normalizada"), pl.col("_marca_alias_norm")]).alias("marca_normalizada")))
    else:
        lf = lf.with_columns(pl.col("_marca_alias_norm").alias("marca_normalizada"))

    # Modelo normalizado (no mapeamos a diccionario)
    lf = lf.with_columns(pl.col("_modelo_norm").alias("modelo_normalizado"))

    # Combustible a 5 buckets con JSON/pares → fallback 'OTROS'
    lf = lf.with_columns(norm_upper(pl.col("combustible")).alias("_comb_alias_norm"))
    if fuels_map is not None and fuels_map.height > 0:
        lf = (lf.join(fuels_map.lazy(), left_on="_comb_alias_norm", right_on="alias_norm", how="left")
                .with_columns(pl.coalesce([pl.col("combustible_right"), pl.lit("OTROS")]).alias("combustible")))
    else:
        lf = lf.with_columns(
            pl.when(pl.col("_comb_alias_norm").str.contains("PHEV|PLUG")).then(pl.lit("PHEV"))
             .when(pl.col("_comb_alias_norm").str.contains("HEV|HIBRIDO|MHEV")).then(pl.lit("HEV"))
             .when(pl.col("_comb_alias_norm").str.contains("ELECTRICO|EV|BEV")).then(pl.lit("BEV"))
             .when(pl.col("_comb_alias_norm").str.contains("DIESEL|GASOIL")).then(pl.lit("DIESEL"))
             .when(pl.col("_comb_alias_norm").str.contains("GASOLINA|PETROL|BENZINA")).then(pl.lit("GASOLINA"))
             .otherwise(pl.lit("OTROS")).alias("combustible")
        )

    # Normalización INE (solo INE, no usamos 'localidad')
    raw_ine = pl.col("codigo_ine").cast(pl.Utf8, strict=False).str.strip_chars().str.replace_all("\r", "").str.replace_all("\n", "")
    is_numeric_like = raw_ine.str.contains(r"^\d+(\.0+)?$", literal=False)
    ine_digits = (
        pl.when(is_numeric_like).then(raw_ine.str.replace(r"\.0+$", "", literal=False)).otherwise(None).cast(pl.Utf8)
    )
    ine_len = ine_digits.str.len_chars()
    codigo_ine_norm = (
        pl.when(ine_len == 4).then(ine_digits.str.pad_start(5, "0"))
         .when(ine_len == 5).then(ine_digits)
         .otherwise(None)
         .alias("codigo_ine")
    )
    lf = lf.with_columns(codigo_ine_norm)

    # Provincia y código provincia (string estable). Si falta, derivar de INE (dos primeros dígitos)
    lf = lf.with_columns([
        norm_upper(pl.col("provincia")).alias("provincia"),
        pl.when(pl.col("codigo_provincia").is_not_null())
          .then(pl.col("codigo_provincia").cast(pl.Utf8))
          .when(pl.col("codigo_ine").is_not_null())
          .then(pl.col("codigo_ine").cast(pl.Utf8).str.slice(0, 2))
          .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("codigo_provincia")
    ])

    # Tipos y tipos auxiliares
    lf = lf.with_columns([
        norm_upper(pl.col("tipo_vehiculo")).alias("tipo_vehiculo_norm"),
        pl.col("transmision").cast(pl.Utf8).alias("transmision"),
        pl.col("antiguedad_anios").cast(pl.Float64, strict=False).alias("antiguedad_anios"),
    ])

    # anio (año de MATRICULACIÓN) con prioridades: año_mat -> fecha_mat -> derivación (yyyymm - antigüedad) -> año(yyyymm)
    # Regla de derivación: anio = floor(yyyymm/100) - round(antiguedad_anios)
    anio_derivado = (
        year_from_yyyymm(pl.col("yyyymm"))
        - pl.col("antiguedad_anios").round(0).cast(pl.Int64)
    )
    lf = lf.with_columns([
        pl.when(pl.col("año_mat").is_not_null())
          .then(pl.col("año_mat").cast(pl.Int64, strict=False))
          .when(pl.col("fecha_mat_parsed").is_not_null())
          .then(pl.col("fecha_mat_parsed").dt.year().cast(pl.Int64))
          .when((pl.col("antiguedad_anios").is_not_null()) & (pl.col("yyyymm").is_not_null()))
          .then(anio_derivado)
          .otherwise(year_from_yyyymm(pl.col("yyyymm")))
          .alias("anio")
    ])

    # Casts finales estables
    lf = lf.with_columns([
        pl.col("yyyymm").cast(pl.Int64, strict=False).alias("yyyymm"),
        pl.col("anio").cast(pl.Int64, strict=False).alias("anio"),
        pl.col("codigo_provincia").cast(pl.Utf8, strict=False).alias("codigo_provincia"),
        pl.col("marca_normalizada").cast(pl.Utf8, strict=False).alias("marca_normalizada"),
        pl.col("modelo_normalizado").cast(pl.Utf8, strict=False).alias("modelo_normalizado"),
        pl.col("combustible").cast(pl.Utf8, strict=False).alias("combustible"),
    ])

    return lf

def keep_tipos(lf: pl.LazyFrame, include: list[str]) -> pl.LazyFrame:
    if not include:
        return lf
    tokens = []
    for t in include:
        u = t.upper().strip()
        u = u.replace("TODOTERRENO", "TODO TERRENO").replace("TODOTERRENOS", "TODO TERRENO")
        u = u.replace("TURISMOS", "TURISMO")
        tokens.append(u)
    pat = "(" + "|".join([re.escape(x) for x in tokens]) + ")"
    return lf.filter(pl.col("tipo_vehiculo_norm").is_null() | pl.col("tipo_vehiculo_norm").str.contains(pat))

