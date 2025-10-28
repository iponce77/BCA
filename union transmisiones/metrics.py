from __future__ import annotations
import polars as pl
from typing import List, Dict, Optional

AGG_KEYS = ["marca_normalizada","modelo_normalizado","anio","combustible","provincia","codigo_provincia","yyyymm"]
AGG_KEYS_INE = ["marca_normalizada","modelo_normalizado","anio","combustible","codigo_ine","yyyymm"]


def _collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)

def yyyymm_add(yyyymm: int, delta: int) -> int:
    y,m = divmod(yyyymm, 100)
    m += delta
    y += (m-1)//12
    m = (m-1)%12 + 1
    return y*100 + m

def aggregate_month(lf: pl.LazyFrame, include_tipos: List[str]) -> pl.DataFrame:
    import dgt_schema as ds
    lf = ds.keep_tipos(lf, include_tipos)
    lf = lf.with_columns(pl.col("yyyymm").cast(pl.Int64, strict=False)).filter(pl.col("yyyymm").is_not_null())
    cnt_0_3 = (pl.col("antiguedad_anios").is_not_null() & (pl.col("antiguedad_anios") <= 3.0)).cast(pl.Int64)
    cnt_4_7 = (pl.col("antiguedad_anios").is_not_null() & (pl.col("antiguedad_anios") > 3.0) & (pl.col("antiguedad_anios") <= 7.0)).cast(pl.Int64)
    cnt_8_m = (pl.col("antiguedad_anios").is_not_null() & (pl.col("antiguedad_anios") > 7.0)).cast(pl.Int64)

    agg = (lf.group_by(AGG_KEYS)
            .agg([
                pl.count().alias("unidades"),
                pl.col("antiguedad_anios").mean().alias("antiguedad_media"),
                pl.col("antiguedad_anios").quantile(0.25).alias("p25_antiguedad"),
                pl.col("antiguedad_anios").median().alias("p50_antiguedad"),
                pl.col("antiguedad_anios").quantile(0.75).alias("p75_antiguedad"),
                cnt_0_3.sum().alias("cnt_0_3"),
                cnt_4_7.sum().alias("cnt_4_7"),
                cnt_8_m.sum().alias("cnt_8_mas"),
            ])
            .with_columns([
                (pl.col("cnt_0_3")*100.0/pl.col("unidades")).fill_null(0.0).round(2).alias("%_0_3"),
                (pl.col("cnt_4_7")*100.0/pl.col("unidades")).fill_null(0.0).round(2).alias("%_4_7"),
                (pl.col("cnt_8_mas")*100.0/pl.col("unidades")).fill_null(0.0).round(2).alias("%_8_mas"),
            ])
            .drop(["cnt_0_3","cnt_4_7","cnt_8_mas"])
    )
    return _collect_streaming(agg)

def add_yoy(df: pl.DataFrame) -> pl.DataFrame:
    prev = df.select(AGG_KEYS + ["unidades"]).with_columns((pl.col("yyyymm")-100).alias("yyyymm")).rename({"unidades":"unidades_prev"})
    out = df.join(prev, on=AGG_KEYS, how="left")
    out = out.with_columns(
        pl.when((pl.col("unidades_prev").is_null()) | (pl.col("unidades_prev")==0))
          .then(None)
          .otherwise(((pl.col("unidades")-pl.col("unidades_prev"))*100.0/pl.col("unidades_prev")).round(2))
          .alias("YoY_unidades_%")
    )
    return out

def add_shares(df: pl.DataFrame) -> pl.DataFrame:
    prov = df.group_by(["provincia","codigo_provincia","yyyymm"]).agg(pl.col("unidades").sum().alias("prov_tot"))
    esp  = df.group_by(["yyyymm"]).agg(pl.col("unidades").sum().alias("esp_tot"))
    out = (df.join(prov, on=["provincia","codigo_provincia","yyyymm"], how="left")
             .join(esp, on=["yyyymm"], how="left")
             .with_columns([
                (pl.col("unidades")*100.0/pl.col("prov_tot")).round(3).alias("share_prov_%"),
                (pl.col("unidades")*100.0/pl.col("esp_tot")).round(3).alias("share_esp_%")
             ])
             .drop(["prov_tot","esp_tot"]))
    return out

def detect_period(df_all: pl.DataFrame, mode: str, year: int | None, months: int | None, end: int | None):
    if df_all.is_empty(): raise ValueError("Agregado vacío tras ETL. Revisa filtros y mapeos.")
    if "yyyymm" not in df_all.columns: raise ValueError("Falta 'yyyymm'. Revisa standardize_lazyframe.")
    df_all = df_all.with_columns(pl.col("yyyymm").cast(pl.Int64, strict=False))
    if df_all.select(pl.col("yyyymm").drop_nulls().len()).item()==0:
        raise ValueError("Todos los yyyymm son nulos.")
    max_mm = int(df_all["yyyymm"].max())
    min_mm = int(df_all["yyyymm"].min())
    if mode=="annual":
        if year is None: raise ValueError("--year requerido en mode=annual")
        start = year*100 + 1
        endc = df_all.filter(pl.col("yyyymm").is_between(start, year*100+12))["yyyymm"]
        if len(endc)==0: raise ValueError(f"No hay meses para {year}")
        return start, int(endc.max())
    elif mode=="rolling":
        if end is None: end = max_mm
        months = months or 12
        # start inclusive: last 'months' months
        y,m = divmod(end,100); import datetime as _dt
        start = end - 100*((months-1)//12) - ((months-1)%12)
        # simple helper via yyyymm_add
        def yyyymm_add(yyyymm, delta): 
            y,m = divmod(yyyymm,100); m+=delta; y+=(m-1)//12; m=(m-1)%12+1; return y*100+m
        start = yyyymm_add(end, -(months-1))
        if start<min_mm: start=min_mm
        return start, end
    else:
        raise ValueError("mode debe ser annual o rolling")

def filter_period(df_all: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    return df_all.filter(pl.col("yyyymm").is_between(start, end))

def summarize(df_all: pl.DataFrame, start: int, end: int, focus_prov: str, top_n: int=20) -> Dict[str, pl.DataFrame]:
    p = filter_period(df_all, start, end)
    kpi = p.select(pl.col("unidades").sum().alias("unidades_total"))
    r_m_es = p.group_by(["marca_normalizada"]).agg(pl.col("unidades").sum().alias("unidades")).sort("unidades", descending=True).head(top_n)
    r_mod_es = p.group_by(["modelo_normalizado"]).agg(pl.col("unidades").sum().alias("unidades")).sort("unidades", descending=True).head(top_n)
    r_prov = p.group_by(["provincia"]).agg(pl.col("unidades").sum().alias("unidades")).sort("unidades", descending=True)
    r_comb = p.group_by(["combustible"]).agg(pl.col("unidades").sum().alias("unidades")).sort("unidades", descending=True)
    focus = focus_prov.upper()
    r_m_focus = p.filter(pl.col("provincia").str.to_uppercase()==focus).group_by(["marca_normalizada"]).agg(pl.col("unidades").sum().alias("unidades")).sort("unidades", descending=True).head(top_n)
    r_mod_focus = p.filter(pl.col("provincia").str.to_uppercase()==focus).group_by(["modelo_normalizado"]).agg(pl.col("unidades").sum().alias("unidades")).sort("unidades", descending=True).head(top_n)
    mix = (p.group_by(["provincia","combustible"]).agg(pl.col("unidades").sum().alias("unidades"))
             .join(p.group_by(["provincia"]).agg(pl.col("unidades").sum().alias("tot")), on="provincia")
             .with_columns((pl.col("unidades")*100.0/pl.col("tot")).round(2).alias("share_prov_%"))
             .drop("tot").sort(["provincia","share_prov_%"], descending=[False, True]))
    return {"kpi":kpi,"rank_marcas_es":r_m_es,"rank_modelos_es":r_mod_es,"rank_provincias":r_prov,"rank_combustible":r_comb,"rank_marcas_focus":r_m_focus,"rank_modelos_focus":r_mod_focus,"mix_comb_prov":mix,"period_df":p}

def df_to_md(df: pl.DataFrame, max_rows: int=10) -> str:
    if df.is_empty(): return "_(sin datos)_"
    cols = df.columns
    rows = df.head(max_rows).to_pandas().values.tolist()
    out = ["|"+"|".join(cols)+"|","|"+"|".join(["---"]*len(cols))+"|"]
    for r in rows:
        out.append("|"+"|".join("" if x is None else str(x) for x in r)+"|")
    return "\n".join(out)

def keys_for_join(period_df: pl.DataFrame, low_support_threshold: int = 10) -> pl.DataFrame:
    return (period_df.group_by(["marca_normalizada","modelo_normalizado","anio","combustible","provincia","codigo_provincia"])
                    .agg(pl.col("unidades").sum().alias("unidades_total_periodo"))
                    .with_columns(pl.when(pl.col("unidades_total_periodo")<low_support_threshold).then(1).otherwise(0).alias("low_support")))


def aggregate_month_ine(lf: pl.LazyFrame, include_tipos: List[str]) -> pl.DataFrame:
    import dgt_schema as ds
    lf = ds.keep_tipos(lf, include_tipos)
    lf = lf.with_columns(pl.col("yyyymm").cast(pl.Int64, strict=False)).filter(pl.col("yyyymm").is_not_null())
    lf = lf.filter(pl.col("codigo_ine").is_not_null())

    cnt_0_3 = (pl.col("antiguedad_anios").is_not_null() & (pl.col("antiguedad_anios") <= 3.0)).cast(pl.Int64)
    cnt_4_7 = (pl.col("antiguedad_anios").is_not_null() & (pl.col("antiguedad_anios") > 3.0) & (pl.col("antiguedad_anios") <= 7.0)).cast(pl.Int64)
    cnt_8_m = (pl.col("antiguedad_anios").is_not_null() & (pl.col("antiguedad_anios") > 7.0)).cast(pl.Int64)

    agg = (lf.group_by(AGG_KEYS_INE)
             .agg([
                 pl.count().alias("unidades"),
                 pl.col("antiguedad_anios").mean().alias("antiguedad_media"),
                 pl.col("antiguedad_anios").quantile(0.25).alias("p25_antiguedad"),
                 pl.col("antiguedad_anios").median().alias("p50_antiguedad"),
                 pl.col("antiguedad_anios").quantile(0.75).alias("p75_antiguedad"),
                 cnt_0_3.sum().alias("cnt_0_3"),
                 cnt_4_7.sum().alias("cnt_4_7"),
                 cnt_8_m.sum().alias("cnt_8_mas"),
             ])
             .with_columns([
                 (pl.col("cnt_0_3")*100.0/pl.col("unidades")).fill_null(0.0).round(2).alias("%_0_3"),
                 (pl.col("cnt_4_7")*100.0/pl.col("unidades")).fill_null(0.0).round(2).alias("%_4_7"),
                 (pl.col("cnt_8_mas")*100.0/pl.col("unidades")).fill_null(0.0).round(2).alias("%_8_mas"),
             ])
             .drop(["cnt_0_3","cnt_4_7","cnt_8_mas"])
    )
    return _collect_streaming(agg)

