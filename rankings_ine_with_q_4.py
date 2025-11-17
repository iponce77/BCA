#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper que añade Q4 sin tocar tu rankings_ine.py original.
- Si se pasa --q4, ejecuta el ranking de mercados (INE/provincia/ccaa) usando rankings_ine_q4.py
- Si NO se pasa --q4, delega 100% en tu rankings_ine.py original (no cambia nada de su flujo)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

# import funciones Q4
from rankings_ine_q4 import load_any, rank_markets, filter_cluster, DEFAULTS, Period


def run_q4(args: argparse.Namespace) -> int:
    # Carga dataset
    df = load_any(args.data)

    # Columnas (permite override desde CLI)
    cols = DEFAULTS.copy()
    if args.col_ine: cols["col_ine"] = args.col_ine
    if args.col_yyyymm: cols["col_yyyymm"] = args.col_yyyymm
    if args.col_brand: cols["col_brand"] = args.col_brand
    if args.col_model: cols["col_model"] = args.col_model
    if args.col_fuel: cols["col_fuel"] = args.col_fuel
    if args.col_units: cols["col_units"] = args.col_units

    # Filtro cluster
    df = filter_cluster(df, args.marca, args.modelo, args.combustible, cols)
    if df.empty:
        print("(sin filas para ese cluster)")
        return 0

    # Periodo
    if args.anio:
        period = Period(int(args.anio) * 100 + 1, int(args.anio) * 100 + 12)
    else:
        period = Period(int(args.start), int(args.end))

    # Ranking
    out = rank_markets(df, period, args.geo, cols, compute_yoy=args.yoy, municipios_path=args.municipios)

    # Salida
    out = out.head(args.top)
    if args.out is None:
        tag = f"{args.marca}_{args.modelo}_{args.geo}_{period.yyyymm_start}-{period.yyyymm_end}"
        args.out = Path(f"q4_markets_{tag}.csv")
    out.to_csv(args.out, index=False)
    print(args.out)
    return 0


def main():
    # Si el usuario incluye --q4, parseamos nuestros flags; si no, delegamos a rankings_ine.py
    if "--q4" not in sys.argv:
        import rankings_ine as legacy
        return legacy.main()  # type: ignore

    ap = argparse.ArgumentParser(description="Q4 – Ranking de mercados (wrapper)")
    ap.add_argument("--q4", action="store_true", help="Activa Q4 (ranking de mercados)")

    # dataset y municipios
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--municipios", type=Path, default=None)

    # cluster
    ap.add_argument("--marca", required=True)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--combustible", nargs="*", default=None)

    # periodo
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--anio", type=int, help="Año completo (YYYY)")
    g.add_argument("--start", type=int, help="Inicio AAAAMM")
    ap.add_argument("--end", type=int, help="Fin AAAAMM")

    # geo/top
    ap.add_argument("--geo", choices=["ine","provincia","ccaa"], default="provincia")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--yoy", action="store_true")

    # columnas (overrides opcionales)
    ap.add_argument("--col-ine", dest="col_ine", default=DEFAULTS["col_ine"])
    ap.add_argument("--col-yyyymm", dest="col_yyyymm", default=DEFAULTS["col_yyyymm"])
    ap.add_argument("--col-brand", dest="col_brand", default=DEFAULTS["col_brand"])
    ap.add_argument("--col-model", dest="col_model", default=DEFAULTS["col_model"])
    ap.add_argument("--col-fuel", dest="col_fuel", default=DEFAULTS["col_fuel"])
    ap.add_argument("--col-units", dest="col_units", default=DEFAULTS["col_units"])

    ap.add_argument("--out", type=Path, default=None)

    args = ap.parse_args()
    return run_q4(args)


if __name__ == "__main__":
    raise SystemExit(main())
