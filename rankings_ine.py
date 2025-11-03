#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rankings_ine.py

Genera rankings por municipio (INE) y periodo a partir de un dataset de transmisiones
en CSV **o** Parquet. Incluye:
- Lectura eficiente:
  * CSV: --chunksize para filtrar en streaming.
  * Parquet: lectura por columnas y por row-group (pyarrow) aplicando filtro temprano.
- Ranking con share, share_top y HHI.
- Mix de combustibles con alias (BEV/HEV/PHEV/MHEV/GLP/GNC/ICE/FCEV/OTROS).
- Comparativa vs periodo previo (--vs-prev).
- Timeseries por yyyymm (--timeseries).
- Metadatos .json junto a cada salida.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None

# ------------------------ Constantes de columnas ------------------------
COL_MARCA = "marca_normalizada"
COL_MODELO = "modelo_normalizado"
COL_COMBUSTIBLE = "combustible"
COL_INE = "codigo_ine"
COL_ANIO = "anio"
COL_YYYYMM = "yyyymm"
COL_UNIDADES = "unidades"

USECOLS = [COL_MARCA, COL_MODELO, COL_COMBUSTIBLE, COL_INE, COL_ANIO, COL_YYYYMM, COL_UNIDADES]

# ------------------------ Alias combustibles ----------------------------
FUEL_ALIASES = {
    "ELECTRICO": "BEV", "ELÉCTRICO": "BEV", "EV": "BEV", "BEV": "BEV",
    "HIBRIDO": "HEV", "HÍBRIDO": "HEV", "HEV": "HEV",
    "HÍBRIDO ENCHUFABLE": "PHEV", "HIBRIDO ENCHUFABLE": "PHEV", "PHEV": "PHEV",
    "MHEV": "MHEV",
    "GLP": "GLP", "GNC": "GNC",
    "GASOLINA": "ICE", "DIESEL": "ICE", "DIÉSEL": "ICE", "GASÓLEO": "ICE",
    "HIDROGENO": "FCEV", "HIDRÓGENO": "FCEV", "FCEV": "FCEV",
}

def fuel_alias(x: str) -> str:
    if not isinstance(x, str):
        return "OTROS"
    key = x.strip().upper()
    return FUEL_ALIASES.get(key, key if key in {"BEV","HEV","PHEV","MHEV","GLP","GNC","ICE","FCEV"} else "OTROS")

# ------------------------ Utilidades municipios -------------------------
def _load_municipios(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"codigo_ine":"Int64"})
    for c in ["municipio","provincia","ccaa"]:
        if c in df.columns:
            df[c+"_norm"] = df[c].astype(str).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.upper().str.strip()
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
    return [int(x) for x in candidates["codigo_ine"].tolist()]

def _period_from_args(args) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if args.anio:
        return int(args.anio), None, None
    if args.start and args.end:
        return None, int(args.start), int(args.end)
    raise SystemExit("Debe indicar --anio o bien --start y --end (yyyymm).")

def _previous_period(anio: Optional[int], start: Optional[int], end: Optional[int]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    # Siempre en años
    if anio is not None:
        return anio - 1, None, None
    # rango previo con misma longitud en años
    length = int(end) - int(start) + 1
    prev_end = int(start) - 1
    prev_start = prev_end - (length - 1)
    return None, prev_start, prev_end


# ------------------------ Carga CSV / Parquet ---------------------------
def _apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    if df.empty:
        return df
    if COL_ANIO not in df.columns:
        raise SystemExit("⛔ El dataset no contiene la columna 'anio', necesaria para el filtrado de periodo.")

    cur = df[df[COL_INE].isin(f.ine)]

    # --- PERIODO: siempre por 'anio' ---
    if f.anio is not None:
        cur = cur[cur[COL_ANIO] == f.anio]
    else:
        # start/end son AÑOS (YYYY) exclusivamente
        sy, ey = int(f.start), int(f.end)
        cur = cur[(cur[COL_ANIO] >= sy) & (cur[COL_ANIO] <= ey)]

    # --- OTROS FILTROS ---
    if f.marca:
        cur = cur[cur[COL_MARCA].astype(str).str.upper() == f.marca.upper()]
    if f.modelo:
        cur = cur[cur[COL_MODELO].astype(str).str.upper() == f.modelo.upper()]
    if f.combustible:
        cur = cur[cur[COL_COMBUSTIBLE].astype(str).str.upper() == f.combustible.upper()]

    for c in [COL_MARCA, COL_MODELO, COL_COMBUSTIBLE]:
        if c in cur.columns:
            cur[c] = cur[c].astype("category")
    return cur


def _iter_csv(csv_path: str, f: Filters, chunksize: Optional[int]) -> pd.DataFrame:
    if chunksize:
        it = pd.read_csv(csv_path, chunksize=chunksize, usecols=USECOLS,
                         dtype={COL_INE:"Int64", COL_ANIO:"Int64", COL_YYYYMM:"Int64"})
        parts = []
        for chunk in it:
            parts.append(_apply_filters(chunk, f))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=USECOLS)
    else:
        df = pd.read_csv(csv_path, usecols=USECOLS,
                         dtype={COL_INE:"Int64", COL_ANIO:"Int64", COL_YYYYMM:"Int64"})
        return _apply_filters(df, f)

def _iter_parquet(pq_path: str, f: Filters) -> pd.DataFrame:
    if pq is None:
        raise SystemExit("pyarrow no está instalado; instálalo para leer Parquet (pip install pyarrow).")
    pf = pq.ParquetFile(pq_path)
    cols = [c for c in USECOLS if c in pf.schema.names]
    parts = []

    def _rg_has(col, pred, rg):
        try:
            cidx = pf.schema.get_field_index(col)
            stats = rg.column(cidx).statistics
            if stats is None:
                return True
            mn, mx = stats.min, stats.max
            return pred(mn, mx)
        except Exception:
            return True

    for i in range(pf.num_row_groups):
        rg = pf.metadata.row_group(i)
        # filtros gruesos por row-group (best-effort)
        passes_period = True
        if f.anio is not None and COL_ANIO in pf.schema.names:
            passes_period = _rg_has(COL_ANIO, lambda mn, mx: (mx >= f.anio and mn <= f.anio), rg)
        elif f.start and f.end and COL_YYYYMM in pf.schema.names:
            s, e = f.start, f.end
            passes_period = _rg_has(COL_YYYYMM, lambda mn, mx: (mx >= s and mn <= e), rg)

        passes_ine = True
        if COL_INE in pf.schema.names:
            ine_min, ine_max = min(f.ine), max(f.ine)
            passes_ine = _rg_has(COL_INE, lambda mn, mx: (mx >= ine_min and mn <= ine_max), rg)

        if not (passes_period and passes_ine):
            continue

        # ← IMPORTANTE: sin types_mapper; conversión estándar y luego coerción manual
        table = pf.read_row_group(i, columns=cols)
        df = table.to_pandas(use_threads=True)

        # Coerción segura a Int64 nullable (evita errores de tipos mixtos)
        for c in (COL_INE, COL_ANIO, COL_YYYYMM, COL_UNIDADES):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

        parts.append(_apply_filters(df, f))

    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)

    # Asegura categorías después de concatenar (por si se perdieron)
    for c in [COL_MARCA, COL_MODELO, COL_COMBUSTIBLE]:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def _iter_filtered(path: str, f: Filters, chunksize: Optional[int]) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return _iter_parquet(path, f)
    elif ext in (".csv", ".gz", ".bz2", ".xz"):
        return _iter_csv(path, f, chunksize)
    else:
        raise SystemExit(f"Formato no soportado para --data: {path}")

# ------------------------ Ranking / Mix / HHI / TS ----------------------
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
    return float((s.sum()) * 10000)

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

def _load_for_period(data_path: str, f: Filters, anio: Optional[int], start: Optional[int], end: Optional[int], chunksize: Optional[int]) -> pd.DataFrame:
    f2 = Filters(ine=f.ine, anio=anio, start=start, end=end,
                 marca=f.marca, modelo=f.modelo, combustible=f.combustible,
                 by_combustible=f.by_combustible, top=0)
    return _iter_filtered(data_path, f2, chunksize)

def _ranking_vs_prev(data_path: str, f: Filters, chunksize: Optional[int], do_vs_prev: bool) -> Tuple[pd.DataFrame, Optional[dict]]:
    cur_df = _iter_filtered(data_path, f, chunksize)
    r = _ranking(cur_df, f)
    meta = None
    if do_vs_prev:
        anio2, s2, e2 = _previous_period(f.anio, f.start, f.end)
        prev_df = _load_for_period(data_path, f, anio2, s2, e2, chunksize)
        r_prev = _ranking(prev_df, f)
        keys = [COL_MARCA, COL_MODELO] + ([COL_COMBUSTIBLE] if f.by_combustible else [])
        merged = r.merge(r_prev[keys+[COL_UNIDADES, "share"]], on=keys, how="left", suffixes=("", "_prev"))
        merged["unidades_prev"] = merged[COL_UNIDADES+"_prev"].fillna(0).astype(int)
        merged["delta_unidades"] = merged[COL_UNIDADES] - merged["unidades_prev"]
        merged["delta_pct"] = merged.apply(lambda row: (row["delta_unidades"]/row["unidades_prev"]*100.0) if row["unidades_prev"]>0 else None, axis=1)
        merged.drop(columns=[COL_UNIDADES+"_prev","share_prev"], inplace=True, errors="ignore")
        r = merged
        meta = {"prev_period": {"anio": anio2, "start": s2, "end": e2},
                "prev_total_unidades": int(prev_df[COL_UNIDADES].sum()) if not prev_df.empty else 0}
    return r, meta, cur_df

# ------------------------ IO / CLI --------------------------------------
def _print_df(df: pd.DataFrame, fmt: str) -> None:
    if df is None or df.empty:
        print("(sin filas)")
        return
    if fmt=="table":
        with pd.option_context('display.max_rows', 200, 'display.max_columns', None, 'display.width', 120):
            print(df.to_string(index=False))
    else:
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
    p = argparse.ArgumentParser(description="Ranking por municipio (INE) y periodo (CSV/Parquet).")
    # municipio
    p.add_argument("--ine", type=int, help="Código INE")
    p.add_argument("--mun", type=str, help="Nombre del municipio")
    p.add_argument("--provincia", type=str)
    p.add_argument("--ccaa", type=str)
    p.add_argument("--on-ambig", choices=["fail","first","all"], default="first")
    # periodo
    p.add_argument("--anio", type=int)
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    # filtros
    p.add_argument("--marca", type=str)
    p.add_argument("--modelo", type=str)
    p.add_argument("--combustible", type=str)
    p.add_argument("--by-combustible", action="store_true")
    p.add_argument("--top", type=int, default=20)
    # IO
    p.add_argument("--data", type=str, default="agg_transmisiones_ine.parquet", help="Ruta a Parquet o CSV")
    p.add_argument("--municipios-csv", type=str, default=os.environ.get("MUNICIPIOS_INE_CSV","data/municipios_ine.csv"))
    p.add_argument("--out-prefix", type=str, default="")
    p.add_argument("--show", action="store_true")
    p.add_argument("--format", choices=["table","csv"], default="table")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--chunksize", type=int, help="Sólo para CSV")
    p.add_argument("--vs-prev", action="store_true")
    p.add_argument("--mix-combustible", action="store_true")
    p.add_argument("--timeseries", action="store_true")

    args = p.parse_args()

    ines = _resolve_ine(args, args.municipios_csv)
    anio, start, end = _period_from_args(args)

    filters = Filters(
        ine=ines, anio=anio, start=start, end=end,
        marca=args.marca, modelo=args.modelo, combustible=args.combustible,
        by_combustible=args.by_combustible, top=args.top
    )

    # Carga datos con el reader correcto por extensión
    df = _iter_filtered(args.data, filters, args.chunksize)

    # Retrocesos si no hay filas
    relax_steps = []
    if df.empty:
        relax_steps.append("combustible")
        tmp_f = Filters(**{**asdict(filters), "combustible": None})
        df = _iter_filtered(args.data, tmp_f, args.chunksize)
        if df.empty and filters.modelo:
            relax_steps.append("modelo")
            tmp_f = Filters(**{**asdict(tmp_f), "modelo": None})
            df = _iter_filtered(args.data, tmp_f, args.chunksize)
        if df.empty and filters.marca:
            relax_steps.append("marca")
            tmp_f = Filters(**{**asdict(tmp_f), "marca": None})
            df = _iter_filtered(args.data, tmp_f, args.chunksize)
        filters = tmp_f

    # Timeseries (opcional)
    ts_df = _timeseries(df, filters) if args.timeseries else pd.DataFrame()

    # Ranking (+ vs-prev)
    ranking, vs_meta, df_used = _ranking_vs_prev(args.data, filters, args.chunksize, args.vs_prev)

    # Métricas
    hhi = _hhi_from_ranking(ranking)
    total_unidades = int(df_used[COL_UNIDADES].sum()) if not df_used.empty else 0

    # Mix
    mix_df = _mix_combustible(df_used) if args.mix_combustible else pd.DataFrame()

    # nombres salida
    period_str = str(anio) if anio is not None else f"{start}-{end}"
    ine_str = "x".join(str(i) for i in ines)
    prefix = args.out_prefix or ""
    rank_out = f"{prefix}ranking_{ine_str}_{period_str}"
    mix_out = f"{prefix}mix_combustible_{ine_str}_{period_str}"
    ts_out = f"{prefix}timeseries_{ine_str}_{period_str}"

    # metadatos
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

    # outputs
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

    # resumen
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
