import polars as pl
import unicodedata
import re

def _strip_accents_upper(s: str) -> str:
    if s is None:
        return s
    # NFD separa letras de “marcas” (acentos); luego quitamos las marcas
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.upper()


COLUMN_SYNONYMS = {
    "fecha_mat": ["fecha_mat","fecha","fecha_matriculacion","fecha matriculación","fecha primera matriculacion","f_mat"],
    "marca": ["make_clean","marca","make","fabricante","brand"],
    "modelo": ["modelo","model","modelo_bare"],
    "submodelo": ["submodelo","version","versión"],
    "vin": ["vin","bastidor","num_bastidor"],
    "combustible": ["combustible","fuel","tipo_combustible"],
    "codigo_ine": ["codigo_ine","cod_ine"],
    "localidad": ["localidad","municipio","poblacion","población"],
    "provincia": ["provincia"],
    "tipo_vehiculo": ["tipo_vehiculo","tipo","categoria","categoría"],
    "transmision": ["transmision","cambio","caja_cambios","caja de cambios"],
    "año_mat": ["año_mat","ano_mat","aÃ±o_mat","anio_mat"],
    "mes_mat": ["mes_mat","mes_matriculacion","mes"],
    "antiguedad_anios": ["antiguedad_anios","antigüedad","antiguedad","edad_anios","edad_anos"],
    "nombre_archivo": ["nombre_archivo","source_file"],
    "es_cruzable": ["es_cruzable","cruzable"],
    "codigo_provincia": ["codigo_provincia","cod_prov","ineprov"],
    # Nota: en tu schema funcional, modelo_base es sinónimo de modelo_normalizado
    "modelo_normalizado": ["modelo_normalizado","modelo_norm","model_normalized","modelo_base"],
}

_ACCENTS = str.maketrans("ÁÀÄÂÃÉÈËÊÍÌÏÎÓÒÖÔÚÙÜÛÑáàäâãéèëêíìïîóòöôúùüûñ",
                         "AAAAAEEEEIIIIOOOOUUUUNAAAAAEEEEIIIIOOOOUUUUN")

def norm_upper(expr: pl.Expr) -> pl.Expr:
    return (
        expr
        .cast(pl.Utf8, strict=False)
        .map_elements(_strip_accents_upper, return_dtype=pl.Utf8)
    )

def build_rename_map_from_names(names: list[str]) -> dict[str,str]:
    lo = [c.lower() for c in names]
    rename={}
    for canon, syns in COLUMN_SYNONYMS.items():
        for s in syns:
            s_low = s.lower()
            if s_low in lo:
                i = lo.index(s_low)
                old = names[i]
                if old != canon: rename[old] = canon
                break
    return rename

def safe_rename_first(lf: pl.LazyFrame) -> pl.LazyFrame:
    # 1) Normaliza nombres crudos (quita espacios y \r/\n)
    names0 = lf.collect_schema().names()
    cleaned = {old: old.strip().replace("\r", "").replace("\n", "")
               for old in names0
               if old != old.strip() or ("\r" in old) or ("\n" in old)}
    if cleaned:
        lf = lf.rename(cleaned)

    # 2) Construye mapa de renombre a canónicos según COLUMN_SYNONYMS
    names = lf.collect_schema().names()
    rm = build_rename_map_from_names(names)

    # 3) Evita duplicados: si el canónico ya existe, NO renombres ese sinónimo
    names_set = set(names)
    rm_safe = {}
    for old, canon in rm.items():
        if canon in names_set:
            # ya existe la columna canónica; saltamos el renombre de este sinónimo
            continue
        rm_safe[old] = canon

    if rm_safe:
        lf = lf.rename(rm_safe)

    return lf


def yyyymm_from_date(col: str) -> pl.Expr:
    return (pl.col(col).dt.year()*100 + pl.col(col).dt.month()).alias("yyyymm")

def standardize_lazyframe(lf: pl.LazyFrame, yyyymm_hint: int | None, brands_map: pl.DataFrame | None, fuels_map: pl.DataFrame | None) -> pl.LazyFrame:
    lf = safe_rename_first(lf)

    # PRIORIDAD: make_clean (si existe) > marca
    names = lf.collect_schema().names()
    if "make_clean" in names:
        # Prioriza make_clean cuando exista
        lf = lf.with_columns(pl.col("make_clean").alias("marca"))

    lf = lf.with_columns(
        pl.col("fecha_mat").cast(pl.Utf8).str.strptime(pl.Date, "%d/%m/%Y", strict=False)
                           .fill_null(pl.col("fecha_mat").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False))
                           .alias("fecha_mat_parsed")
    )

    hint_year = int(yyyymm_hint)//100 if yyyymm_hint else None

    # --- anio = AÑO DE MATRICULACIÓN (prioridad: año_mat -> fecha_mat_parsed -> hint_year)
    lf = lf.with_columns([
        pl.when(pl.col("año_mat").is_not_null())
          .then(pl.col("año_mat").cast(pl.Int64, strict=False))
          .when(pl.col("fecha_mat_parsed").is_not_null())
          .then(pl.col("fecha_mat_parsed").dt.year())
          .otherwise(pl.lit(hint_year, dtype=pl.Int64))
          .alias("anio"),
    ])

    # --- yyyymm = MES DE TRANSMISIÓN (prioridad: 'transmision' "YYYY-MM"/"YYYY/MM" -> yyyymm_hint)
    lf = lf.with_columns([
        pl.when(
            pl.col("transmision").is_not_null()
            & pl.col("transmision").cast(pl.Utf8, strict=False)
              .str.contains(r"^\d{4}[-/]\d{2}$", literal=False)
        ).then(
            pl.col("transmision").cast(pl.Utf8, strict=False)
              .str.slice(0, 7)                               # "YYYY-MM" o "YYYY/MM"
              .str.replace_all(r"[-/]", "", literal=False)   # "YYYYMM"
              .cast(pl.Int64, strict=False)
        ).otherwise(
            pl.lit(yyyymm_hint, dtype=pl.Int64)
        ).alias("yyyymm"),
    ])

    # --- NORMALIZACIÓN DE INE (solo INE, sin localidad) ---
    # Reglas:
    # - Si es numérico como "28065" o "28065.0" => usar 28065
    # - Si tiene 4 dígitos => pad a 5 con '0' delante (p.ej. 4561 -> 04561)
    # - Si no es numérico => null (no rompe)
    raw_ine = pl.col("codigo_ine").cast(pl.Utf8, strict=False).str.strip_chars().str.replace_all("\r", "").str.replace_all("\n", "")
    is_numeric_like = raw_ine.str.contains(r"^\d+(\.0+)?$", literal=False)
    ine_digits = (
        pl.when(is_numeric_like)
          .then(raw_ine.str.replace(r"\.0+$", "", literal=False))
          .otherwise(None)
          .cast(pl.Utf8)
    )
    ine_len = ine_digits.str.len_chars()
    codigo_ine_norm = (
        pl.when(ine_len == 4).then(ine_digits.str.pad_start(5, "0"))
         .when(ine_len == 5).then(ine_digits)
         .otherwise(None)
         .alias("codigo_ine")
    )
    lf = lf.with_columns(codigo_ine_norm)

    lf = lf.with_columns([
        pl.col("provincia").cast(pl.Utf8).str.strip_chars().alias("provincia"),
        pl.when(pl.col("codigo_provincia").is_not_null())
          .then(pl.col("codigo_provincia").cast(pl.Utf8))
          .when(pl.col("codigo_ine").is_not_null())
          .then(pl.col("codigo_ine").cast(pl.Utf8).str.slice(0,2))
          .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("codigo_provincia")
    ])

    lf = lf.with_columns([
        norm_upper(pl.col("tipo_vehiculo")).alias("tipo_vehiculo_norm"),
        pl.col("transmision").cast(pl.Utf8).alias("transmision"),
        pl.col("antiguedad_anios").cast(pl.Float64, strict=False).alias("antiguedad_anios"),
    ])

    lf = lf.with_columns(norm_upper(pl.col("marca")).alias("_marca_alias_norm"))
    if brands_map is not None and brands_map.height > 0:
        lf = (lf.join(brands_map.lazy(), left_on="_marca_alias_norm", right_on="alias_norm", how="left")
                .with_columns(pl.coalesce([pl.col("marca_normalizada"), norm_upper(pl.col("marca"))]).alias("marca_normalizada")))
    else:
        lf = lf.with_columns(norm_upper(pl.col("marca")).alias("marca_normalizada"))

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

    # PRIORIDAD: modelo_base > modelo_normalizado (existente) > modelo
    names = lf.collect_schema().names()
    if "modelo_base" in names:
        # Si existe modelo_base en el esquema, priorizamos su uso (cuando aporte texto)
        trimmed_modelo_base = norm_upper(pl.col("modelo_base"))
        if "modelo_normalizado" in names:
            trimmed_modelo_norm = norm_upper(pl.col("modelo_normalizado"))
            lf = lf.with_columns(
                pl.when(pl.col("modelo_base").is_not_null() & (trimmed_modelo_base.str.len_chars() > 0))
                  .then(pl.col("modelo_base").cast(pl.Utf8))
                  .when(pl.col("modelo_normalizado").is_not_null() & (trimmed_modelo_norm.str.len_chars() > 0))
                  .then(pl.col("modelo_normalizado").cast(pl.Utf8))
                  .otherwise(pl.lit(None, dtype=pl.Utf8))
                  .alias("modelo_normalizado")
            )
        else:
            lf = lf.with_columns(
                pl.when(pl.col("modelo_base").is_not_null() & (trimmed_modelo_base.str.len_chars() > 0))
                  .then(pl.col("modelo_base").cast(pl.Utf8))
                  .otherwise(pl.lit(None, dtype=pl.Utf8))
                  .alias("modelo_normalizado")
            )
    elif "modelo_normalizado" in names:
        # No hay modelo_base, pero ya existe modelo_normalizado: usarlo si aporta valor
        trimmed_modelo_norm = norm_upper(pl.col("modelo_normalizado"))
        lf = lf.with_columns(
            pl.when(pl.col("modelo_normalizado").is_not_null() & (trimmed_modelo_norm.str.len_chars() > 0))
              .then(pl.col("modelo_normalizado").cast(pl.Utf8))
              .otherwise(pl.lit(None, dtype=pl.Utf8))
              .alias("modelo_normalizado")
        )
    else:
        lf = lf.with_columns(pl.lit(None, dtype=pl.Utf8).alias("modelo_normalizado"))
   

    lf = lf.with_columns([
        pl.col("yyyymm").cast(pl.Int64, strict=False).alias("yyyymm"),
        pl.col("anio").cast(pl.Int64, strict=False).alias("anio"),
    ])
    return lf

def keep_tipos(lf: pl.LazyFrame, include: list[str]) -> pl.LazyFrame:
    if not include: return lf
    tokens=[]
    for t in include:
        u = t.upper().strip()
        u = u.replace("TODOTERRENO","TODO TERRENO").replace("TODOTERRENOS","TODO TERRENO")
        u = u.replace("TURISMOS","TURISMO")
        tokens.append(u)
    pat = "(" + "|".join([re.escape(x) for x in tokens]) + ")"
    return lf.filter(pl.col("tipo_vehiculo_norm").is_null() | pl.col("tipo_vehiculo_norm").str.contains(pat))
