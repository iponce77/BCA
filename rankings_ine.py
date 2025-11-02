#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rankings_ine.py

Genera rankings de vehículos (marca+modelo) por municipio (codigo_ine) y periodo, a partir de agg_transmisiones_ine.csv.

Mejoras añadidas:
- --chunksize: lectura en chunks y filtrado temprano por INE/periodo para reducir memoria.
- Tipos category para columnas de texto tras normalizar.
- --format table|csv para la salida por consola (--show).
- Exporta metadatos .json junto a cada CSV con filtros y resumen.
- --dry-run: no escribe ficheros, solo muestra conteos y filtros resueltos.
- share_top: columna con % acumulado del ranking.
- HHI: índice de concentración del top-N.
- --vs-prev: compara vs el periodo anterior equivalente (anio-1 o rango desplazado) y añade Δuds/Δ%.
- Alias de combustibles (BEV/HEV/PHEV/MHEV/GLP/GNC/ICE) para el mix.
- --timeseries: exporta series mensuales por marca+modelo (si hay yyyymm en datos).

Columnas esperadas en agg_transmisiones_ine.csv:
    marca_normalizada, modelo_normalizado, anio, yyyymm, combustible, codigo_ine, unidades
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

# ------------------------ Constantes de columnas ------------------------
COL_MARCA = "marca_normalizada"
COL_MODELO = "modelo_normalizado"
COL_COMBUSTIBLE = "combustible"
COL_INE = "codigo_ine"
COL_ANIO = "anio"
COL_YYYYMM = "yyyymm"
COL_UNIDADES = "unidades"

# ------------------------ Alias combustibles ----------------------------
FUEL_ALIASES = {
    # eléctricos
    "ELECTRICO": "BEV",
    "ELÉCTRICO": "BEV",
    "EV": "BEV",
    "BEV": "BEV",
    # híbridos
    "HIBRIDO": "HEV",
    "HÍBRIDO": "HEV",
    "HEV": "HEV",
    "HÍBRIDO ENCHUFABLE": "PHEV",
    "HIBRIDO ENCHUFABLE": "PHEV",
    "PHEV": "PHEV",
    "MHEV": "MHEV",
    # gaseosos
    "GLP": "GLP",
    "GNC": "GNC",
    # térmicos
    "GASOLINA": "ICE",
    "DIESEL": "ICE",
    "DIÉSEL": "ICE",
    "GASÓLEO": "ICE",
    "HIDROGENO": "FCEV",
    "HIDRÓGENO": "FCEV",
    "FCEV": "FCEV",
}

def fuel_alias(x: str) -> str:
    if not isinstance(x, str):
        return "OTROS"
    key = x.strip().upper()
    return FUEL_ALIASES.get(key, key if key in {"BEV","HEV","PHEV","MHEV","GLP","GNC","ICE","FCEV"} else "OTROS")

# ------------------------ Utilidades ------------------------------------
def _load_municipios(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"codigo_ine":"Int64"})
    # normalizaciones básicas
    for c in ["municipio","provincia","ccaa"]:
        if c in df.columns:
            df[c+"_norm"] = df[c].str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.upper().str.strip()
        else:
            df[c+"_norm"] = ""
    return df

def _resolve_ine(args, municipios_path: str) -> List[int]:
    if args.ine:
        return [int(args.ine)]
    df = _load_municipios(municipios_path)
    q = (args.mun or "").strip()
    if not q:
        raise SystemExit("Debe indicar --ine o --mun.")
    qn = (pd.Series([q]).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.upper().str.strip()).iloc[0]
    cur = df
    if args.provincia:
        pn = (pd.Series([args.provincia]).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.upper().str.strip()).iloc[0]
        cur = cur[cur["provincia_norm"]==pn]
    if args.ccaa:
        cn = (pd.Series([args.ccaa]).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.upper().str.strip()).iloc[0]
        cur = cur[cur["ccaa_norm"]==cn]
    candidates = cur[cur["municipio_norm"]==qn]
    if len(candidates)==0:
        # empieza por
        candidates = cur[cur["municipio_norm"].str.startswith(qn)]
    if len(candidates)==0:
        candidates = cur[cur["municipio_norm"].str.contains(qn, na=False)]
    if len(candidates)==0:
        raise SystemExit(f"No se encontró municipio '{q}'. Pruebe con --provincia o --ccaa.")
    if args.on_ambig=="fail" and len(candidates)>1:
        tops = candidates.head(10)[["municipio","provincia","codigo_ine"]].to_dict(orient="records")
        raise SystemExit(f"Múltiples municipios coinciden: {tops}. Use --provincia o --ccaa, o cambie --on-ambig.")
    if args.on_ambig=="first":
        return [int(candidates.iloc[0]["codigo_ine"])]
    # all
    return [int(x) for x in candidates["codigo_ine"].tolist()]

def _period_from_args(args) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """devuelve (anio, start, end)"""
    if args.anio:
        return int(args.anio), None, None
    if args.start and args.end:
        return None, int(args.start), int(args.end)
    raise SystemExit("Debe indicar --anio o bien --start y --end (yyyymm).")

def _previous_period(anio: Optional[int], start: Optional[int], end: Optional[int]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if anio is not None:
        return anio-1, None, None
    # desplaza el rango por su longitud
    # asume formato yyyymm
    s, e = str(start), str(end)
    sy, sm = int(s[:4]), int(s[4:])
    ey, em = int(e[:4]), int(e[4:])
    # longitud en meses
    length = (ey - sy)*12 + (em - sm) + 1
    # rango anterior: termina el mes anterior a start
    em2 = sm - 1
    ey2 = sy
    if em2 == 0:
        em2 = 12
        ey2 -= 1
    # inicio = final - (length-1) meses
    total_prev_end = ey2*12 + em2
    total_prev_start = total_prev_end - (length-1)
    psy, psm = divmod(total_prev_start, 12)
    if psm==0:
        psy -= 1; psm = 12
    pstart = int(f"{psy:04d}{psm:02d}")
    pend = int(f"{ey2:04d}{em2:02d}")
    return None, pstart, pend

@dataclass
class Filters:
    ine: List[int]
    anio: Optional[int]
    start: Optional[int]
    end: Optional[int]
    marca: Optional[str]
    modelo: Optional[str]
    combustible: Optional[str]
    by_combustible: bool
    top: int

# ------------------------ Carga con chunks y filtrado temprano ----------
def _iter_filtered(csv_path: str, filters: Filters, chunksize: Optional[int]) -> pd.DataFrame:
    usecols = [COL_MARCA, COL_MODELO, COL_COMBUSTIBLE, COL_INE, COL_ANIO, COL_YYYYMM, COL_UNIDADES]
    if chunksize:
        it = pd.read_csv(csv_path, chunksize=chunksize, usecols=usecols, dtype={COL_INE:"Int64", COL_ANIO:"Int64", COL_YYYYMM:"Int64"})
        parts = []
        for chunk in it:
            parts.append(_apply_filters(chunk, filters))
        if len(parts)==0:
            return pd.DataFrame(columns=usecols)
        return pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(csv_path, usecols=usecols, dtype={COL_INE:"Int64", COL_ANIO:"Int64", COL_YYYYMM:"Int64"})
        return _apply_filters(df, filters)

def _apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    df = df[df[COL_INE].isin(f.ine)]
    if f.anio is not None:
        df = df[df[COL_ANIO]==f.anio]
    else:
        df = df[(df[COL_YYYYMM]>=f.start) & (df[COL_YYYYMM]<=f.end)]
    if f.marca:
        df = df[df[COL_MARCA].str.upper()==f.marca.upper()]
    if f.modelo:
        df = df[df[COL_MODELO].str.upper()==f.modelo.upper()]
    if f.combustible:
        df = df[df[COL_COMBUSTIBLE].str.upper()==f.combustible.upper()]
    # tipificar
    for c in [COL_MARCA, COL_MODELO, COL_COMBUSTIBLE]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

# ------------------------ Ranking, HHI y mix ----------------------------
def _ranking(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [COL_MARCA, COL_MODELO] + ([COL_COMBUSTIBLE] if f.by_combustible else [])
    g = df.groupby(keys, observed=True, sort=False, dropna=False)[COL_UNIDADES].sum().reset_index()
    g = g.sort_values(COL_UNIDADES, ascending=False)
    g["share"] = (g[COL_UNIDADES] / g[COL_UNIDADES].sum()) * 100.0
    g["share_top"] = g["share"].cumsum()
    g["rank"] = range(1, len(g)+1)
    if f.top:
        g = g.head(f.top)
    return g

def _hhi_from_ranking(r: pd.DataFrame) -> float:
    if r.empty:
        return 0.0
    s = (r["share"]/100.0) ** 2
    return float((s.sum()) * 10000)  # 0..10000

def _mix_combustible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    base = df.copy()
    base["comb_alias"] = base[COL_COMBUSTIBLE].map(fuel_alias)
    g = base.groupby("comb_alias", observed=True)[COL_UNIDADES].sum().reset_index()
    g = g.sort_values(COL_UNIDADES, ascending=False)
    total = g[COL_UNIDADES].sum()
    g["share"] = (g[COL_UNIDADES] / total) * 100.0 if total else 0.0
    return g

def _timeseries(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    if df.empty or COL_YYYYMM not in df.columns:
        return pd.DataFrame()
    keys = [COL_YYYYMM, COL_MARCA, COL_MODELO]
    if f.by_combustible:
        keys.append(COL_COMBUSTIBLE)
    g = df.groupby(keys, observed=True)[COL_UNIDADES].sum().reset_index()
    return g.sort_values([COL_MARCA, COL_MODELO, COL_YYYYMM])

# ------------------------ Comparación vs periodo previo ------------------
def _load_for_period(csv_path: str, ine: List[int], anio: Optional[int], start: Optional[int], end: Optional[int], chunksize: Optional[int], f: Filters) -> pd.DataFrame:
    f2 = Filters(ine=ine, anio=anio, start=start, end=end, marca=f.marca, modelo=f.modelo,
                 combustible=f.combustible, by_combustible=f.by_combustible, top=0)
    return _iter_filtered(csv_path, f2, chunksize)

def _ranking_vs_prev(csv_path: str, f: Filters, chunksize: Optional[int]) -> Tuple[pd.DataFrame, Optional[dict]]:
    cur_df = _iter_filtered(args.csv, f, chunksize)
    r = _ranking(cur_df, f)
    meta = None
    if args.vs_prev:
        anio2, s2, e2 = _previous_period(f.anio, f.start, f.end)
        prev_df = _load_for_period(csv_path, f.ine, anio2, s2, e2, chunksize, f)
        r_prev = _ranking(prev_df, f)
        # agregar por claves
        keys = [COL_MARCA, COL_MODELO] + ([COL_COMBUSTIBLE] if f.by_combustible else [])
        merged = r.merge(r_prev[keys+[COL_UNIDADES, "share"]], on=keys, how="left", suffixes=("", "_prev"))
        merged["unidades_prev"] = merged[COL_UNIDADES+"_prev"].fillna(0).astype(int)
        merged["delta_unidades"] = merged[COL_UNIDADES] - merged["unidades_prev"]
        merged["delta_pct"] = merged.apply(lambda row: (row["delta_unidades"]/row["unidades_prev"]*100.0) if row["unidades_prev"]>0 else None, axis=1)
        merged.drop(columns=[COL_UNIDADES+"_prev","share_prev"], inplace=True, errors="ignore")
        r = merged
        meta = {
            "prev_period": {"anio": anio2, "start": s2, "end": e2},
            "prev_total_unidades": int(prev_df[COL_UNIDADES].sum()) if not prev_df.empty else 0
        }
    return r, meta

# ------------------------ Main ------------------------------------------
def _print_df(df: pd.DataFrame, fmt: str) -> None:
    if df is None or df.empty:
        print("(sin filas)")
        return
    if fmt=="table":
        # anchuras automáticas
        with pd.option_context('display.max_rows', 200, 'display.max_columns', None, 'display.width', 120):
            print(df.to_string(index=False))
    else:
        # csv
        print(df.to_csv(index=False))

def _save_with_meta(df: pd.DataFrame, out_path: str, meta: dict, dry: bool):
    base = os.path.splitext(out_path)[0]
    csv_path = f"{base}.csv"
    json_path = f"{base}.json"
    if not dry:
        df.to_csv(csv_path, index=False)
        with open(json_path,"w",encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return csv_path, json_path

def main():
    global args
    p = argparse.ArgumentParser(description="Ranking por municipio (INE) y periodo.")
    # selección de municipio
    p.add_argument("--ine", type=int, help="Código INE")
    p.add_argument("--mun", type=str, help="Nombre del municipio (alternativa a --ine)")
    p.add_argument("--provincia", type=str, help="Desambiguación por provincia")
    p.add_argument("--ccaa", type=str, help="Desambiguación por CCAA")
    p.add_argument("--on-ambig", choices=["fail","first","all"], default="first", help="Qué hacer si el nombre es ambiguo.")
    # periodo
    p.add_argument("--anio", type=int, help="Año (YYYY)")
    p.add_argument("--start", type=int, help="Inicio (yyyymm)")
    p.add_argument("--end", type=int, help="Fin (yyyymm)")
    # filtros
    p.add_argument("--marca", type=str)
    p.add_argument("--modelo", type=str)
    p.add_argument("--combustible", type=str)
    p.add_argument("--by-combustible", action="store_true")
    p.add_argument("--top", type=int, default=20)
    # IO
    p.add_argument("--csv", type=str, default="agg_transmisiones_ine.csv")
    p.add_argument("--municipios-csv", type=str, default=os.environ.get("MUNICIPIOS_INE_CSV","data/municipios_ine.csv"))
    p.add_argument("--out-prefix", type=str, default="")
    p.add_argument("--show", action="store_true")
    p.add_argument("--format", choices=["table","csv"], default="table", help="Formato para --show.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--chunksize", type=int, help="Tamaño de chunk para lectura eficiente.")
    p.add_argument("--vs-prev", action="store_true", help="Comparar contra periodo previo equivalente.")
    p.add_argument("--mix-combustible", action="store_true", help="Generar mix de combustibles.")
    p.add_argument("--timeseries", action="store_true", help="Exportar serie mensual por marca+modelo.")

    args = p.parse_args()

    # Resolver INE(s)
    ines = _resolve_ine(args, args.municipios_csv)
    anio, start, end = _period_from_args(args)

    filters = Filters(
        ine=ines, anio=anio, start=start, end=end,
        marca=args.marca, modelo=args.modelo, combustible=args.combustible,
        by_combustible=args.by_combustible, top=args.top
    )

    # Carga datos
    df = _iter_filtered(args.csv, filters, args.chunksize)

    # Retrocesos si no hay filas
    relax_steps = []
    if df.empty:
        relax_steps.append("combustible")
        tmp_f = Filters(**{**asdict(filters), "combustible": None})
        df = _iter_filtered(args.csv, tmp_f, args.chunksize)
        if df.empty and filters.modelo:
            relax_steps.append("modelo")
            tmp_f = Filters(**{**asdict(tmp_f), "modelo": None})
            df = _iter_filtered(args.csv, tmp_f, args.chunksize)
        if df.empty and filters.marca:
            relax_steps.append("marca")
            tmp_f = Filters(**{**asdict(tmp_f), "marca": None})
            df = _iter_filtered(args.csv, tmp_f, args.chunksize)
        filters = tmp_f

    # Timeseries (opcional)
    ts_df = _timeseries(df, filters) if args.timeseries else pd.DataFrame()

    # Ranking y vs-prev
    ranking, vs_meta = _ranking_vs_prev(args.csv, filters, args.chunksize)

    # Métricas
    hhi = _hhi_from_ranking(ranking)
    total_unidades = int(df[COL_UNIDADES].sum()) if not df.empty else 0

    # Mix
    mix_df = _mix_combustible(df) if args.mix_combustible else pd.DataFrame()

    # nombres de salida
    period_str = str(anio) if anio is not None else f"{start}-{end}"
    ine_str = "x".join(str(i) for i in ines)
    prefix = args.out_prefix or ""
    rank_out = f"{prefix}ranking_{ine_str}_{period_str}"
    mix_out = f"{prefix}mix_combustible_{ine_str}_{period_str}"
    ts_out = f"{prefix}timeseries_{ine_str}_{period_str}"

    # Metadatos comunes
    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "ine": ines,
        "period": {"anio": anio, "start": start, "end": end},
        "filters": {
            "marca": filters.marca, "modelo": filters.modelo, "combustible": filters.combustible,
            "by_combustible": filters.by_combustible, "top": filters.top
        },
        "relaxations": relax_steps,
        "total_unidades": total_unidades,
        "hhi_top": hhi
    }
    if vs_meta:
        metadata["vs_prev"] = vs_meta

    # Salidas
    outputs = []
    if not ranking.empty:
        csv_path, json_path = _save_with_meta(ranking, rank_out, metadata, args.dry_run)
        outputs += [csv_path, json_path]
        if args.show:
            _print_df(ranking, args.format)
    else:
        print("No hay datos para el filtro/periodo (tras retrocesos: %s)." % "→".join(relax_steps or ["ninguno"]), file=sys.stderr)

    if args.mix_combustible and not mix_df.empty:
        _, _ = _save_with_meta(mix_df, mix_out, metadata, args.dry_run)
        outputs.append(mix_out + ".csv")

    if args.timeseries and not ts_df.empty:
        _, _ = _save_with_meta(ts_df, ts_out, metadata, args.dry_run)
        outputs.append(ts_out + ".csv")

    # Resumen
    print(json.dumps({
        "outputs": outputs,
        "summary": {
            "ine": ines, "period": period_str, "rows": int(len(ranking)),
            "total_unidades": total_unidades, "hhi_top": hhi,
            "relaxations": relax_steps
        }
    }, ensure_ascii=False))

if __name__ == "__main__":
    raise SystemExit(main())
