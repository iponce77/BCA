#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q4 – ¿Dónde se vende mejor este modelo/año?

Ranking de mercados (INE / provincia / CCAA) para un *cluster* concreto
(marca + modelo_base_x|modelo + combustible opcional) y un periodo dado.

Métrica principal: unidades totales en el periodo por mercado.
Métricas de apoyo: share del mercado, YoY, coef_var_pct (estabilidad mensual) y HHI de marcas.

Diseñado para ser plug-and-play y flexible en nombres de columnas.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

# --------------------------- Defaults de columnas ---------------------------
DEFAULTS = {
    "col_ine": "codigo_ine",           # código INE del municipio (int/str)
    "col_yyyymm": "yyyymm",            # AAAAMM (int/str)
    "col_brand": "marca",              # marca normalizada
    "col_model": "modelo_base_x",      # modelo canónico (fallback a modelo)
    "col_fuel": "combustible_norm",    # combustible normalizado (opcional)
    "col_units": "unidades",           # nº transmisiones/matriculaciones
}

# --------------------------- Utilidades ------------------------------------

def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.upper().str.strip()

@dataclass
class Period:
    yyyymm_start: int
    yyyymm_end: int

    @property
    def length_months(self) -> int:
        ys, ms = divmod(self.yyyymm_start, 100)
        ye, me = divmod(self.yyyymm_end, 100)
        return (ye - ys) * 12 + (me - ms) + 1

    def previous(self) -> "Period":
        # periodo previo contiguo de igual longitud
        length = self.length_months
        y, m = divmod(self.yyyymm_start, 100)
        # end of previous period = month before start
        py, pm = y, m - 1
        if pm == 0:
            py -= 1; pm = 12
        pend = py * 100 + pm
        # start = pend - (length-1) months
        sy, sm = py, pm
        for _ in range(length - 1):
            sm -= 1
            if sm == 0:
                sy -= 1; sm = 12
        pstart = sy * 100 + sm
        return Period(pstart, pend)

# --------------------------- Carga y mapeos --------------------------------

def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)

def attach_geo(df: pd.DataFrame, municipios_path: Optional[Path], col_ine: str) -> pd.DataFrame:
    if municipios_path is None:
        return df
    if not municipios_path.exists():
        return df
    mun = load_any(municipios_path)
    # Intentamos columnas estándar
    # Se esperan campos: codigo_ine, municipio, provincia, ccaa
    candidates = {
        "codigo_ine": ["codigo_ine", "ine", "cod_ine", "codmun"],
        "municipio": ["municipio", "mun", "localidad"],
        "provincia": ["provincia", "prov"],
        "ccaa": ["ccaa", "ca", "comunidad_autonoma"],
    }
    def pick(col: str) -> Optional[str]:
        for c in candidates[col]:
            if c in mun.columns:
                return c
        return None
    key = pick("codigo_ine")
    if not key:
        return df
    keep = {
        "codigo_ine": key,
        "municipio": pick("municipio"),
        "provincia": pick("provincia"),
        "ccaa": pick("ccaa"),
    }
    mun = mun[[v for v in keep.values() if v is not None]].copy()
    mun = mun.rename(columns={keep["codigo_ine"]: "codigo_ine",
                              keep.get("municipio", "municipio"): "municipio",
                              keep.get("provincia", "provincia"): "provincia",
                              keep.get("ccaa", "ccaa"): "ccaa"})
    # normaliza tipos
    mun["codigo_ine"] = _to_int(mun["codigo_ine"])  # para merge robusto

    out = df.copy()
    out["__ine_int"] = _to_int(out[col_ine])
    out = out.merge(mun, left_on="__ine_int", right_on="codigo_ine", how="left")
    out = out.drop(columns=["__ine_int"])  # limpiar
    return out

# --------------------------- Core Q4 ---------------------------------------

def rank_markets(df: pd.DataFrame,
                 period: Period,
                 geo_level: str,
                 cols: Dict[str, str],
                 compute_yoy: bool = True,
                 municipios_path: Optional[Path] = None) -> pd.DataFrame:
    """Devuelve ranking de mercados para el periodo."""
    col_ine = cols["col_ine"]; col_ym = cols["col_yyyymm"]; col_units = cols["col_units"]; col_brand = cols["col_brand"]
    # Attach geo labels
    df = attach_geo(df, municipios_path, col_ine)

    # Filtro periodo actual
    df = df[(pd.to_numeric(df[col_ym], errors="coerce") >= period.yyyymm_start) &
            (pd.to_numeric(df[col_ym], errors="coerce") <= period.yyyymm_end)]

    # Determina columna geo
    geo_level = geo_level.lower()
    if geo_level == "ine":
        geo_col = col_ine
    elif geo_level == "provincia":
        geo_col = "provincia"
    elif geo_level == "ccaa":
        geo_col = "ccaa"
    else:
        raise ValueError("geo debe ser ine|provincia|ccaa")

    # Agregado de unidades por mercado
    g = (df.groupby(geo_col, observed=True)[col_units]
            .sum()
            .reset_index()
            .rename(columns={geo_col: "geo", col_units: "unidades"}))
    g = g.sort_values("unidades", ascending=False)
    total = g["unidades"].sum()
    g["share_pct"] = (g["unidades"] / total * 100.0) if total else 0.0

    # CV mensual por mercado
    ts = (df.groupby([geo_col, col_ym], observed=True)[col_units]
            .sum().reset_index().rename(columns={geo_col: "geo"}))
    cv = (ts.groupby("geo")[col_units].agg(["mean", "std"]).reset_index())
    cv["coef_var_pct"] = (cv["std"] / cv["mean"] * 100.0).replace([np.inf, -np.inf], np.nan)
    g = g.merge(cv[["geo", "coef_var_pct"]], on="geo", how="left")

    # HHI de marcas por mercado
    parts = []
    for geo, chunk in df.groupby(geo_col, observed=True):
        r = chunk.groupby(col_brand)[col_units].sum().reset_index()
        s = r[col_units].sum()
        if s <= 0:
            hhi = np.nan
        else:
            r["share"] = r[col_units] / s
            hhi = float((r["share"] ** 2).sum() * 10000)  # 0..10000
        parts.append({"geo": geo, "hhi_marca": hhi})
    g = g.merge(pd.DataFrame(parts), on="geo", how="left")

    # YoY (periodo previo contiguo
    if compute_yoy:
        prev = period.previous()
        df_prev = df[(pd.to_numeric(df[col_ym], errors="coerce") >= prev.yyyymm_start) &
                     (pd.to_numeric(df[col_ym], errors="coerce") <= prev.yyyymm_end)]
        prev_g = (df_prev.groupby(geo_col, observed=True)[col_units]
                        .sum().reset_index().rename(columns={geo_col: "geo", col_units: "unidades_prev"}))
        g = g.merge(prev_g, on="geo", how="left")
        g["unidades_prev"] = g["unidades_prev"].fillna(0).astype(int)
        g["yoy_unidades"] = g["unidades"] - g["unidades_prev"]
        g["yoy_pct"] = np.where(g["unidades_prev"] > 0, g["yoy_unidades"] / g["unidades_prev"] * 100.0, np.nan)
    else:
        g["yoy_unidades"] = np.nan
        g["yoy_pct"] = np.nan

    # Orden recomendado (primario): unidades desc, yoy_pct desc, coef_var asc, hhi asc
    g = g.sort_values(["unidades", "yoy_pct", "coef_var_pct", "hhi_marca"], ascending=[False, False, True, True])
    return g

# --------------------------- Filtros de cluster ----------------------------

def filter_cluster(df: pd.DataFrame,
                   marca: Optional[str],
                   modelo: Optional[str],
                   combustible: Optional[str | List[str]],
                   cols: Dict[str, str]) -> pd.DataFrame:
    col_brand = cols["col_brand"]; col_model = cols["col_model"]; col_fuel = cols["col_fuel"]
    out = df.copy()
    if marca:
        out[col_brand] = _clean_str(out[col_brand])
        marca = _clean_str(pd.Series([marca]))[0]
        out = out[out[col_brand].str.contains(marca, na=False)]
    if modelo:
        # usa modelo_base_x si existe; si no, cae a "modelo"
        if col_model not in out.columns and "modelo" in out.columns:
            col_model = "modelo"
        out[col_model] = _clean_str(out[col_model])
        modelo = _clean_str(pd.Series([modelo]))[0]
        out = out[out[col_model].str.contains(modelo, na=False)]
    if combustible is not None and col_fuel in out.columns:
        if isinstance(combustible, str):
            combustible = [combustible]
        fuels = {_clean_str(pd.Series([f])).iloc[0] for f in combustible}
        out[col_fuel] = _clean_str(out[col_fuel])
        out = out[out[col_fuel].isin(fuels)]
    return out

# --------------------------- CLI ------------------------------------------

def parse_period(args: argparse.Namespace) -> Period:
    if args.anio is not None:
        y = int(args.anio)
        return Period(y * 100 + 1, y * 100 + 12)
    if args.start is not None and args.end is not None:
        return Period(int(args.start), int(args.end))
    raise ValueError("Debes indicar --anio o --start y --end (AAAAMM)")


def main():
    ap = argparse.ArgumentParser(description="Q4 – Ranking de mercados para un cluster marca+modelo(+combustible)")
    ap.add_argument("--data", required=True, type=Path, help="Fichero INE (parquet/csv/xlsx) con columnas de unidades por yyyymm y mercado")
    ap.add_argument("--municipios", type=Path, default=None, help="CSV/Parquet de municipios para mapear codigo_ine→municipio/provincia/ccaa")

    # cluster
    ap.add_argument("--marca", required=True, help="Marca a filtrar (match contiene, case-insensitive)")
    ap.add_argument("--modelo", required=True, help="Modelo a filtrar (match contiene, case-insensitive)")
    ap.add_argument("--combustible", nargs="*", default=None, help="Combustible opcional (uno o varios)")

    # periodo
    ap.add_argument("--anio", type=int, default=None, help="Año completo (ej. 2022)")
    ap.add_argument("--start", type=int, default=None, help="Inicio AAAAMM")
    ap.add_argument("--end", type=int, default=None, help="Fin AAAAMM")
    ap.add_argument("--yoy", action="store_true", help="Calcular YoY vs periodo previo contiguo")

    # geo
    ap.add_argument("--geo", choices=["ine", "provincia", "ccaa"], default="provincia")
    ap.add_argument("--top", type=int, default=20, help="Top-N de mercados")

    # nombres de columnas (por si difieren)
    ap.add_argument("--col-ine", default=DEFAULTS["col_ine"]) 
    ap.add_argument("--col-yyyymm", default=DEFAULTS["col_yyyymm"]) 
    ap.add_argument("--col-brand", default=DEFAULTS["col_brand"]) 
    ap.add_argument("--col-model", default=DEFAULTS["col_model"]) 
    ap.add_argument("--col-fuel", default=DEFAULTS["col_fuel"]) 
    ap.add_argument("--col-units", default=DEFAULTS["col_units"]) 

    ap.add_argument("--out", type=Path, default=None, help="Ruta CSV de salida (si no, se genera nombre automático)")

    args = ap.parse_args()

    # Carga
    df = load_any(args.data)

    # Renombrado flexible si hace falta
    cols = {
        "col_ine": args.col_ine,
        "col_yyyymm": args.col_yyyymm,
        "col_brand": args.col_brand,
        "col_model": args.col_model,
        "col_fuel": args.col_fuel,
        "col_units": args.col_units,
    }
    for need in cols.values():
        if need not in df.columns:
            # intento de fallback de modelo
            if need == args.col_model and "modelo" in df.columns:
                cols["col_model"] = "modelo"
            else:
                raise SystemExit(f"Falta la columna requerida: {need}")

    # Filtro cluster
    df = filter_cluster(df, args.marca, args.modelo, args.combustible, cols)
    if df.empty:
        raise SystemExit("No hay filas para ese cluster (marca/modelo/combustible)")

    # Periodo
    period = parse_period(args)

    # Ranking
    g = rank_markets(df, period, args.geo, cols, compute_yoy=args.yoy, municipios_path=args.municipios)

    # Orden final + Top
    g = g.head(args.top)

    # Si el geo es INE y tenemos nombres, añadimos nombres para legibilidad
    if args.geo == "ine" and {"municipio", "provincia", "ccaa"}.issubset(set(g.columns)):
        # ya vienen de attach_geo si municipios no es None
        pass

    # Salida
    if args.out is None:
        tag = f"{args.marca}_{args.modelo}_{args.geo}_{period.yyyymm_start}-{period.yyyymm_end}"
        fname = f"q4_markets_{tag}.csv"
        args.out = Path(fname)
    g.to_csv(args.out, index=False)
    print(args.out)

if __name__ == "__main__":
    main()

        }
    }, ensure_ascii=False))

if __name__ == "__main__":
    raise SystemExit(main())
