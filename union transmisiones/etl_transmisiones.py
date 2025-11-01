from __future__ import annotations
import argparse, os, logging, json, gzip  # gzip queda por si hay .zip; no usamos .csv.gz
from pathlib import Path
import polars as pl

import utils_io as uio
import dgt_schema as ds
import metrics as mx
import mappings_loader as ml


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ETL + análisis de transmisiones DGT (prod v3, parquet-only)")
    p.add_argument("--input-dir", type=str, default=None)
    p.add_argument("--input-files", type=str, nargs="*", default=None)
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--mode", type=str, choices=["annual", "rolling"], required=True)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--months", type=int, default=None)  # <- si None: procesa todos

    # Defaults quirúrgicos según petición:
    # - mappings en repo: BCA/union transmisiones/mappings/bca_mappings.yml
    # - fuel_aliases.json en raíz del repo
    # Dado que este archivo está en .../BCA/union transmisiones/, el directorio actual (THIS_DIR)
    # es .../union transmisiones. La raíz del repo es THIS_DIR.parent.parent (…/).
    THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = THIS_DIR.parent.parent  # .../<repo_root>
    DEFAULT_MAPPINGS = THIS_DIR / "mappings" / "bca_mappings.yml"
    DEFAULT_FUEL = REPO_ROOT / "fuel_aliases.json"

    p.add_argument("--mappings-file", type=str, default=str(DEFAULT_MAPPINGS))
    p.add_argument("--fuel-json", type=str, default=str(DEFAULT_FUEL))
    p.add_argument("--whitelist-xlsx", type=str, default=None)

    p.add_argument("--include-tipo-vehiculo", type=str, default="")
    p.add_argument("--focus-prov", type=str, default="")
    p.add_argument("--low-support-threshold", type=int, default=10)

    p.add_argument("--preflight-only", action="store_true", default=False)
    p.add_argument("--auto-snapshots", action="store_true", default=False)

    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


# ---------------------------
# Utilidades locales del ETL
# ---------------------------
def preflight(paths: list[str]) -> dict:
    """
    Diagnóstico rápido:
    - Si es .parquet -> lo reporta como tal.
    - Si es .zip -> indica que se inspeccionará el interior buscando .parquet.
    (No se leen .csv/.csv.gz)
    """
    for p in paths:
        lp = p.lower()
        if lp.endswith(".parquet"):
            return {"path": p, "type": "parquet"}
        if lp.endswith(".zip"):
            return {"path": p, "type": "zip (se buscarán .parquet dentro)"}
    return {"note": "No .parquet/.zip found for preflight"}


def pick_last_n_by_month(paths: list[str], n: int | None) -> tuple[list[str], list[int]]:
    """
    Selecciona los últimos N archivos por YYYYMM inferido del NOMBRE.
    - Si n es None o <=0 -> devuelve todos.
    - Desduplica por mes: si hay varios archivos del mismo YYYYMM, se queda con el último.
    """
    pairs = []
    for p in paths:
        mm = uio.infer_yyyymm_from_name(p)
        if mm:
            pairs.append((p, int(mm)))
    if not pairs:
        return [], []
    pairs.sort(key=lambda x: x[1])
    by_month: dict[int, str] = {}
    for p, mm in pairs:
        by_month[mm] = p
    months_sorted = sorted(by_month.keys())
    if not months_sorted:
        return [], []
    picked_months = months_sorted[-n:] if (n and n > 0) else months_sorted
    picked_paths = [by_month[mm] for mm in picked_months]
    return picked_paths, picked_months


# ---------------------------
# Export por periodo + report
# ---------------------------
def export_for_period(
    df_all: pl.DataFrame,
    start: int,
    end: int,
    args,
    out_dir: str,
    df_all_ine: pl.DataFrame | None = None
):
    os.makedirs(out_dir, exist_ok=True)

    df_period = mx.filter_period(df_all, start, end)
    if df_period.is_empty():
        raise SystemExit(f"Sin datos en periodo {start}→{end}.")

    # Ficheros principales (ESP)
    df_period.write_parquet(os.path.join(out_dir, "agg_transmisiones.parquet"))
    df_period.write_csv(os.path.join(out_dir, "agg_transmisiones.csv"))

    # INE (si se facilitó)
    if df_all_ine is not None and "yyyymm" in df_all_ine.columns:
        df_ine_period = mx.filter_period(df_all_ine, start, end)
        if not df_ine_period.is_empty():
            df_ine_period.write_parquet(os.path.join(out_dir, "agg_transmisiones_ine.parquet"))
            df_ine_period.write_csv(os.path.join(out_dir, "agg_transmisiones_ine.csv"))

    # Rankings y resúmenes
    try:
        summary = mx.summarize(df_all, start, end, focus_prov=args.focus_prov, top_n=20)
        if "rank_marcas_es" in summary:
            summary["rank_marcas_es"].write_csv(os.path.join(out_dir, "rank_marcas.csv"))
        if "rank_modelos_es" in summary:
            summary["rank_modelos_es"].write_csv(os.path.join(out_dir, "rank_modelos.csv"))
        if "rank_provincias" in summary:
            summary["rank_provincias"].write_csv(os.path.join(out_dir, "rank_provincias.csv"))
        if "rank_combustible" in summary:
            summary["rank_combustible"].write_csv(os.path.join(out_dir, "rank_combustible.csv"))
        if "mix_comb_prov" in summary:
            summary["mix_comb_prov"].write_csv(os.path.join(out_dir, "mix_combustible_provincia.csv"))
        if "period_df" in summary:
            summary["period_df"].write_parquet(os.path.join(out_dir, "keys_for_join.parquet"))
    except Exception as e:
        logging.warning("No se pudo generar el informe/resúmenes: %s", e)


# ---------------------------
# Proceso principal (SOLO PARQUET)
# ---------------------------
def process_all(args):
    uio.setup_logging(args.log_level)

    # Filtrar explícitamente a parquet/zip desde la enumeración de entrada
    all_paths = uio.iter_input_files(args.input_dir, args.input_files)
    paths = [p for p in all_paths if p.lower().endswith((".parquet", ".zip"))]
    if not paths:
        raise SystemExit("No hay ficheros de entrada .parquet o .zip (con .parquet dentro).")

    logging.info(f"Preflight: {preflight(paths)}")

    selected_months_for_files: list[int] = []
    if args.mode == "rolling":
        paths, selected_months_for_files = pick_last_n_by_month(paths, args.months if args.months else None)
        if not paths:
            raise SystemExit("No se pudieron inferir meses de los nombres de archivo. Esperado YYYYMM en el nombre.")
        logging.info(
            "Procesando archivos por meses en nombre: %s",
            f"{selected_months_for_files[0]}–{selected_months_for_files[-1]}" if selected_months_for_files else "(todos)"
        )

    # Diccionarios (mappings + fuel desde rutas pedidas)
    maps = ml.load_all(args.mappings_file, args.fuel_json, args.whitelist_xlsx)
    logging.info(f"Diccionarios: brands={maps['brands_n']} fuels={maps['fuels_n']}")

    tmp_dir = os.path.join(args.out_dir, "_tmp_zip")
    os.makedirs(tmp_dir, exist_ok=True)

    agg_months: list[pl.DataFrame] = []
    agg_months_ine: list[pl.DataFrame] = []
    months_seen: list[int | None] = []
    include_tipos = [s.strip() for s in (args.include_tipo_vehiculo or "").split(",") if s.strip()]

    # --- Ingesta de cada archivo seleccionado (solo parquet o zip con parquet) ---
    for p in paths:
        lp = p.lower()

        if lp.endswith(".zip"):
            for inner in uio.unzip_to_tmp(p, tmp_dir):
                if not inner.lower().endswith(".parquet"):
                    continue
                m = uio.infer_yyyymm_from_name(inner) or uio.infer_yyyymm_from_name(p)
                months_seen.append(m)

                lf_raw = pl.scan_parquet(inner)
                lf = ds.standardize_lazyframe(
                    lf_raw,
                    yyyymm_hint=m,
                    brands_map=maps["brands_df"],
                    fuels_map=maps["fuels_df"]
                )
                agg_months.append(mx.aggregate_month(lf, include_tipos))
                agg_months_ine.append(mx.aggregate_month_ine(lf, include_tipos))

        elif lp.endswith(".parquet"):
            m = uio.infer_yyyymm_from_name(p)
            months_seen.append(m)

            lf_raw = pl.scan_parquet(p)
            lf = ds.standardize_lazyframe(
                lf_raw,
                yyyymm_hint=m,
                brands_map=maps["brands_df"],
                fuels_map=maps["fuels_df"]
            )
            agg_months.append(mx.aggregate_month(lf, include_tipos))
            agg_months_ine.append(mx.aggregate_month_ine(lf, include_tipos))

        else:
            # Por diseño no llegamos aquí: sólo .parquet/.zip
            continue

    if not agg_months:
        raise SystemExit("No se generaron agregados (ver filtros/entradas).")

    # Concat agregados
    df_all = pl.concat(agg_months, how="vertical")
    df_all_ine = pl.concat(agg_months_ine, how="vertical") if agg_months_ine else None
    if df_all.is_empty():
        raise SystemExit("Agregado vacío tras ETL. Revisa filtros y mappings.")

    # Enriquecimientos post-agg
    df_all = mx.add_yoy(df_all)
    df_all = mx.add_shares(df_all)

    # Validaciones mínimas
    required = ["marca_normalizada","modelo_normalizado","anio","combustible","provincia","codigo_provincia","yyyymm","unidades"]
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise SystemExit(f"Faltan columnas requeridas en el agregado: {missing}")

    # Preflight-only
    if args.preflight_only:
        rep = {
            "meses_detectados": sorted([int(x) for x in months_seen if x]),
            "filas_agg": int(df_all.height),
            "yyyymm_min": int(df_all["yyyymm"].min()),
            "yyyymm_max": int(df_all["yyyymm"].max()),
        }
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "preflight_report.json"), "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        uio.cleanup_tmp(tmp_dir)
        return

    # Selección de periodo y export
    if args.mode == "annual":
        year = int(args.year) if args.year else int(df_all["yyyymm"].max()) // 100
        start = year * 100 + 1
        end = year * 100 + 12
        if df_all.filter(pl.col("yyyymm").is_between(start, end)).height == 0:
            raise ValueError(f"No hay meses para {year}")
        export_for_period(df_all, start, end, args, os.path.join(args.out_dir, f"{year}"), df_all_ine=df_all_ine)

    elif args.mode == "rolling":
        if selected_months_for_files:
            start = int(min(selected_months_for_files))
            end = int(max(selected_months_for_files))
        else:
            start = int(df_all["yyyymm"].min())
            end = int(df_all["yyyymm"].max())
        export_for_period(df_all, start, end, args, os.path.join(args.out_dir, f"rolling_{args.months or 'all'}m"), df_all_ine=df_all_ine)

    else:
        raise ValueError("Mode no reconocido.")

    uio.cleanup_tmp(tmp_dir)


def main():
    args = parse_args()
    process_all(args)


if __name__ == "__main__":
    main()
