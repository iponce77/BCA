# bca_enrich_all.py
# -*- coding: utf-8 -*-
"""
CLI único: ejecuta matching strict→relax (con checkpoints) + enriquecimiento + análisis.
Salida:
- bca_enriched.xlsx (pestaña bca_enriched)
- bca_enriched_analysis.xlsx (roi_por_modelo, cobertura_por_tipo, flags_ratio, top50_margin_abs, top50_margin_pct, y resumen_fallos si disponible)
- audit_matching.xlsx (Top-K por fila; incluye filas "unmatched")
- checkpoints/bca_matched.parquet (reutilizable con --skip-matching-if-exists)
"""
import argparse
import logging
from pathlib import Path
import pandas as pd

from bca_enrich_lib import (
    setup_logging,
    load_config,
    run_matching,
    run_enrichment,
)

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _write_audit_excel(audit_df: pd.DataFrame, out_path: Path) -> None:
    _ensure_parent(out_path)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        audit_df.to_excel(xw, index=False, sheet_name="audit_topk")

def _build_resumen_fallos_from_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    # Sólo filas 'unmatched' con diagnóstico
    df = audit_df.copy()
    df = df[df["decision"].fillna("")=="unmatched"]
    cols = [c for c in ["exact_fail_step","approx_fail_step"] if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["fail_source","fail_step","count"])
    long = []
    for _, r in df.iterrows():
        for src in ["exact","approx"]:
            key = f"{src}_fail_step"
            if key in df.columns:
                fs = r.get(key)
                if pd.notna(fs):
                    long.append({"fail_source": src.upper(), "fail_step": fs})
    if not long:
        return pd.DataFrame(columns=["fail_source","fail_step","count"])
    ll = pd.DataFrame(long)
    return ll.groupby(["fail_source","fail_step"]).size().reset_index(name="count").sort_values("count", ascending=False)

def main():
    ap = argparse.ArgumentParser(description="BCA Enrich ALL: matching strict→relax + enriquecimiento + auditoría + checkpoints")
    ap.add_argument("--config", required=True, help="Ruta YAML de configuración")
    ap.add_argument("--skip-matching-if-exists", action="store_true", help="Reutiliza checkpoint si existe")
    ap.add_argument("--checkpoint-dir", default=None, help="Carpeta para checkpoints (por defecto, la del YAML o 'checkpoints')")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    # Entradas / salidas (desde YAML)
    inputs = cfg.get("inputs", {})
    outputs = cfg.get("outputs", {})
    columns = cfg.get("columns", {})
    enrich_params = (cfg.get("enrich_params") or {})

    bca_path = Path(inputs.get("bca_norm") or inputs.get("bca") or "bca_norm.xlsx")
    master_path = Path(inputs.get("master") or inputs.get("ganvam") or "ganvam.xlsx")

    # Checkpoints (Parquet)
    cp_dir = Path(args.checkpoint_dir or outputs.get("checkpoint_dir") or "checkpoints")
    cp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(outputs.get("checkpoint_match") or (cp_dir / "bca_matched.parquet"))
    if checkpoint_path.is_dir():  # si en YAML dieron carpeta, usar archivo por defecto dentro
        checkpoint_path = checkpoint_path / "bca_matched.parquet"
    if checkpoint_path.parent != cp_dir:
        # honrar path del YAML, pero asegurando carpeta existente
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Paths de salida (Excel existentes)
    bca_enriched_xlsx = Path(outputs.get("bca_enriched_xlsx") or "bca_enriched.xlsx")
    analysis_xlsx = Path(outputs.get("analysis_xlsx") or "bca_enriched_analysis.xlsx")
    audit_xlsx = Path(outputs.get("audit_xlsx") or "audit_matching.xlsx")

    # 1) Matching strict→relax (o checkpoint)
    if args.skip_matching_if_exists and checkpoint_path.exists():
        logging.info("Saltando matching: cargando checkpoint %s", checkpoint_path)
        matches = pd.read_parquet(checkpoint_path)
        audit_df = None
    else:
        logging.info("Ejecutando matching strict→relax...")
        matches, audit_df = run_matching(str(bca_path), str(master_path), cfg)
        # guardar checkpoint
        _ensure_parent(checkpoint_path)
        matches.to_parquet(checkpoint_path, index=False)
        logging.info("Checkpoint guardado en %s", checkpoint_path)

    # 2) Auditoría
    if audit_df is None:
        logging.info("No se generó auditoría (viene de checkpoint). Se omitirá audit_matching.xlsx.")
    else:
        _write_audit_excel(audit_df, audit_xlsx)
        logging.info("Auditoría Top-K: %s", audit_xlsx)

    # 3) Enriquecimiento + análisis
    logging.info("Ejecutando enriquecimiento...")
    enriched, sheets = run_enrichment(str(bca_path), matches, str(master_path), enrich_params)

    # Añadir resumen_fallos (desde audit si lo hay) a las hojas de análisis
    if audit_df is not None:
        resumen = _build_resumen_fallos_from_audit(audit_df)
        if not resumen.empty:
            sheets["resumen_fallos"] = resumen

    # 4) Escribir salidas
    # 4.a) Enriched: mantenemos Excel y añadimos Parquet (salida final en Parquet)
    logging.info("Escribiendo %s", bca_enriched_xlsx)
    _ensure_parent(bca_enriched_xlsx)
    with pd.ExcelWriter(bca_enriched_xlsx, engine="openpyxl") as xw:
        # nombre de hoja fijo para no romper flujos posteriores
        enriched.to_excel(xw, index=False, sheet_name="bca_enriched")

    enriched_parquet = bca_enriched_xlsx.with_suffix(".parquet")
    logging.info("Escribiendo también Parquet → %s", enriched_parquet)
    _ensure_parent(enriched_parquet)
    enriched.to_parquet(enriched_parquet, index=False)

    # 4.b) Análisis (multi-hoja) y audit en Excel, como hasta ahora
    logging.info("Escribiendo análisis %s", analysis_xlsx)
    _ensure_parent(analysis_xlsx)
    with pd.ExcelWriter(analysis_xlsx, engine="openpyxl") as xw:
        for name, sdf in sheets.items():
            sdf.to_excel(xw, index=False, sheet_name=name)

    logging.info("Listo.")

if __name__ == "__main__":
    main()
