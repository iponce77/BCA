from __future__ import annotations
import argparse, os, logging, json, gzip
import polars as pl

import utils_io as uio
import dgt_schema as ds
import metrics as mx
import mappings_loader as ml


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ETL + análisis de transmisiones DGT (prod v3)")
    p.add_argument("--input-dir", type=str, default=None)
    p.add_argument("--input-files", type=str, nargs="*", default=None)
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--mode", type=str, choices=["annual", "rolling"], required=True)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--months", type=int, default=None)  # <- si None: no se limita por N, procesa todos

    p.add_argument("--mappings-file", type=str, default=None)
    p.add_argument("--fuel-json", type=str, default=None)
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
    """Lee cabecera y primera fila de un .csv.gz si existe, para diagnóstico rápido."""
    for p in paths:
        if p.lower().endswith(".csv.gz"):
            sep, enc = ",", "utf-8"
            try:
                with gzip.open(p, "rb") as gz:
                    header = gz.readline().decode(enc, errors="replace").rstrip("\n")
                    row1 = gz.readline().decode(enc, errors="replace").rstrip("\n")
                return {
                    "path": p,
                    "sep": sep,
                    "enc": enc,
                    "header": header.split(sep),
                    "rows": [row1.split(sep)],
                }
            except Exception as e:
                return {"path": p, "error": f"{type(e).__name__}: {e}"}
        if p.lower().endswith(".parquet"):
            return {"path": p, "type": "parquet"}
    return {"note": "No .csv.gz/.parquet found for preflight"}


def hdr_keep_from_file(path: str, sep: str) -> list[str]:
    """Mantiene la cabecera original pero elimina únicamente 'MARCA'/'MODELO' (duplicadas malas)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        first = f.readline().replace("\r", "").replace("\n", "")
    hdr = [h.strip() for h in first.split(sep) if h.strip()]
    return [c for c in hdr if c not in ("MARCA", "MODELO")]


def pick_last_n_by_month(paths: list[str], n: int | None) -> tuple[list[str], list[int]]:
    """
    Selecciona los últimos N archivos por YYYYMM inferido del NOMBRE.
    - Si n es None o <=0 -> devuelve todos los meses disponibles.
    - Desduplica por mes: si hay varios archivos del mismo YYYYMM, se queda con el último en orden de aparición.
    """
    pairs = []
    for p in paths:
        mm = uio.infer_yyyymm_from_name(p)
        if mm:
            pairs.append((p, int(mm)))
    if not pairs:
        return [], []
    # Orden ascendente por YYYYMM y nos quedamos con el último archivo de cada mes
    pairs.sort(key=lambda x: x[1])
    by_month: dict[int, str] = {}
    for p, mm in pairs:
        by_month[mm] = p  # sobreescribe, así el último archivo por mes prevalece
    months_sorted = sorted(by_month.keys())
    if not months_sorted:
        return [], []
    if n and n > 0:
        picked_months = months_sorted[-n:]
    else:
        picked_months = months_sorted
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
    # CSV disabled: official output is Parquet
    # df_period.write_csv(os.path.join(out_dir, "agg_transmisiones.csv"))

    # INE (si se facilitó)
    if df_all_ine is not None and "yyyymm" in df_all_ine.columns:
        df_ine_period = mx.filter_period(df_all_ine, start, end)
        if not df_ine_period.is_empty():
            df_ine_period.write_parquet(os.path.join(out_dir, "agg_transmisiones_ine.parquet"))
             # CSV disabled: official output is Parquet
            # df_ine_period.write_csv(os.path.join(out_dir, "agg_transmisiones_ine.csv"))

        # Rankings y resúmenes
        try:
            summary = mx.summarize(
                df_all, start, end,
                focus_prov=args.focus_prob if hasattr(args, "focus_prob") else args.focus_prov,
                top_n=20
            )
            out_rank_marcas      = os.path.join(out_dir, "rank_marcas.parquet")
            out_rank_modelos     = os.path.join(out_dir, "rank_modelos.parquet")
            out_rank_provincias  = os.path.join(out_dir, "rank_provincias.parquet")
            out_rank_combustible = os.path.join(out_dir, "rank_combustible.parquet")
            out_mix_comb_prov    = os.path.join(out_dir, "mix_combustible_provincia.parquet")

            # Ranks (España) en PARQUET
            if "rank_marcas_es" in summary:
                summary["rank_marcas_es"].write_parquet(out_rank_marcas)
            if "rank_modelos_es" in summary:
                summary["rank_modelos_es"].write_parquet(out_rank_modelos)
            if "rank_provincias" in summary:
                summary["rank_provincias"].write_parquet(out_rank_provincias)
            if "rank_combustible" in summary:
                summary["rank_combustible"].write_parquet(out_rank_combustible)
            if "mix_comb_prov" in summary:
                summary["mix_comb_prov"].write_parquet(out_mix_comb_prov)

            # Las keys para joins ya estaban en Parquet (lo mantenemos)
            if "period_df" in summary:
                summary["period_df"].write_parquet(os.path.join(out_dir, "keys_for_join.parquet"))
        except Exception as e:
            logging.warning("No se pudo generar el informe/resúmenes: %s", e)



# ---------------------------
# Proceso principal
# ---------------------------
def process_all(args):

    # Resolve external paths from repo root (parent of current file's directory)
    try:
        from pathlib import Path
        REPO_ROOT = Path(__file__).resolve().parent.parent
        if args.mappings_file is None:
            args.mappings_file = str(REPO_ROOT / "union transmisiones" / "mappings" / "bca_mappings.yml")
        if args.fuel_json is None:
            # Prefer mappings/fuels.json (legacy). Si no existe, usar fuel_aliases.json en repo root.
            default_fuel = REPO_ROOT / "mappings/fuels.json"
            if not default_fuel.exists():
                default_fuel = REPO_ROOT / "fuel_aliases.json"
            args.fuel_json = str(default_fuel) if default_fuel.exists() else None
        if args.whitelist_xlsx is None:
            default_wl = REPO_ROOT / "whitelist.xlsx"
            args.whitelist_xlsx = str(default_wl) if default_wl.exists() else None
    except Exception:
        pass
    uio.setup_logging(args.log_level)
    paths = uio.iter_input_files(args.input_dir, args.input_files)
    if not paths:
        raise SystemExit("No hay ficheros de entrada (.zip/.csv.gz/.csv/.parquet)")

    logging.info(f"Preflight: {preflight(paths)}")

    # Si mode=rolling y se especifica months (o incluso si es None, queremos selección por archivos):
    selected_months_for_files: list[int] = []
    if args.mode == "rolling":
        # Si months es None -> procesa todos los archivos disponibles (no recorte por N)
        paths, selected_months_for_files = pick_last_n_by_month(paths, args.months if args.months else None)
        if not paths:
            raise SystemExit("No se pudieron inferir meses de los nombres de archivo. Esperado YYYYMM en el nombre.")
        logging.info(
            "Procesando archivos por meses en nombre: %s",
            f"{selected_months_for_files[0]}–{selected_months_for_files[-1]}" if selected_months_for_files else "(todos)"
        )

    # Diccionarios
    maps = ml.load_all(args.mappings_file, args.fuel_json, args.whitelist_xlsx)
    logging.info(f"Diccionarios: brands={maps['brands_n']} fuels={maps['fuels_n']}")

    tmp_dir = os.path.join(args.out_dir, "_tmp_utf8")
    os.makedirs(tmp_dir, exist_ok=True)

    agg_months: list[pl.DataFrame] = []
    agg_months_ine: list[pl.DataFrame] = []
    months_seen: list[int | None] = []
    include_tipos = [s.strip() for s in (args.include_tipo_vehiculo or "").split(",") if s.strip()]

    # --- Ingesta de cada archivo seleccionado ---
    for p in paths:
        if p.lower().endswith(".zip"):
            for inner in uio.unzip_to_tmp(p, tmp_dir):
                m = uio.infer_yyyymm_from_name(inner) or uio.infer_yyyymm_from_name(p)
                months_seen.append(m)

                if inner.lower().endswith(".parquet"):
                    lf_raw = pl.scan_parquet(inner)
                elif inner.lower().endswith(".csv.gz"):
                    csv_path, sep, enc = uio.transcode_gz_to_utf8_tmp(inner, tmp_dir)
                    lf0 = pl.scan_csv(csv_path, separator=sep, infer_schema_length=8000, ignore_errors=True)
                    cols = hdr_keep_from_file(csv_path, sep)
                    lf_raw = lf0.select(cols)
                elif inner.lower().endswith(".csv"):
                    lf0 = pl.scan_csv(inner, separator=",", infer_schema_length=8000, ignore_errors=True)
                    cols = hdr_keep_from_file(inner, ",")
                    lf_raw = lf0.select(cols)
                else:
                    continue

                lf = ds.standardize_lazyframe(lf_raw, yyyymm_hint=m, brands_map=maps["brands_df"], fuels_map=maps["fuels_df"])
                agg_months.append(mx.aggregate_month(lf, include_tipos))
                agg_months_ine.append(mx.aggregate_month_ine(lf, include_tipos))

        elif p.lower().endswith(".parquet"):
            m = uio.infer_yyyymm_from_name(p); months_seen.append(m)
            lf_raw = pl.scan_parquet(p)
            lf = ds.standardize_lazyframe(lf_raw, yyyymm_hint=m, brands_map=maps["brands_df"], fuels_map=maps["fuels_df"])
            agg_months.append(mx.aggregate_month(lf, include_tipos))
            agg_months_ine.append(mx.aggregate_month_ine(lf, include_tipos))

        elif p.lower().endswith(".csv.gz"):
            m = uio.infer_yyyymm_from_name(p); months_seen.append(m)
            csv_path, sep, enc = uio.transcode_gz_to_utf8_tmp(p, tmp_dir)
            lf0 = pl.scan_csv(csv_path, separator=sep, infer_schema_length=8000, ignore_errors=True)
            cols = hdr_keep_from_file(csv_path, sep)
            lf_raw = lf0.select(cols)
            lf = ds.standardize_lazyframe(lf_raw, yyyymm_hint=m, brands_map=maps["brands_df"], fuels_map=maps["fuels_df"])
            agg_months.append(mx.aggregate_month(lf, include_tipos))
            agg_months_ine.append(mx.aggregate_month_ine(lf, include_tipos))

        elif p.lower().endswith(".csv"):
            m = uio.infer_yyyymm_from_name(p); months_seen.append(m)
            lf0 = pl.scan_csv(p, separator=",", infer_schema_length=8000, ignore_errors=True)
            cols = hdr_keep_from_file(p, ",")
            lf_raw = lf0.select(cols)
            lf = ds.standardize_lazyframe(lf_raw, yyyymm_hint=m, brands_map=maps["brands_df"], fuels_map=maps["fuels_df"])
            agg_months.append(mx.aggregate_month(lf, include_tipos))
            agg_months_ine.append(mx.aggregate_month_ine(lf, include_tipos))

        else:
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
        # Hemos preseleccionado los archivos (y meses) mediante pick_last_n_by_month
        if selected_months_for_files:
            start = int(min(selected_months_for_files))
            end = int(max(selected_months_for_files))
        else:
            # Fallback por si no se pudo inferir del nombre (procesa todos los archivos)
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



