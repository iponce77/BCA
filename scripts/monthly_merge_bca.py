#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly merger for BCA daily Excels.

- Lee ficheros .xlsx en --indir que cumplan --pattern (por defecto '*_completo.xlsx').
- Extrae la fecha del nombre (YYYYMMDD) o usa mtime como fallback.
- Concatena, calcula vin_repeat (# de apariciones por VIN en el mes),
  y se queda con la **última** aparición por VIN (por fecha de archivo).
- Escribe **Parquet** (bca_norm_YYYY-MM.parquet). Opcionalmente ejecuta normalizacionv2.py
  para reforzar make_clean/modelo_base, etc., sobre el mismo Parquet.

Uso:
  python scripts/monthly_merge_bca.py \
    --indir monthly_inputs \
    --out bca_norm_2025-08.parquet \
    --normalizador normalizacionv2.py \
    --whitelist whitelist.xlsx \
    --pattern "*_completo.xlsx"     # (opcional, por defecto)
"""
import argparse
import re
from pathlib import Path
from datetime import datetime
import pandas as pd


# Acepta YYYYMMDD en cualquier parte del nombre
DATE_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")

# Sinónimos de VIN frecuentes
VIN_SYNONYMS = ["vin", "VIN", "Vin", "Bastidor", "bastidor", "BASTIDOR"]


def parse_file_date(p: Path) -> datetime:
    m = DATE_RE.search(p.stem)
    if m:
        y, mo, d = map(int, m.groups())
        return datetime(y, mo, d)
    # fallback a mtime si no hay fecha en el nombre
    return datetime.fromtimestamp(p.stat().st_mtime)


def choose_vin_column(df: pd.DataFrame) -> str | None:
    # Busca una columna VIN (aceptando sinónimos). Prioriza 'vin' exacta si existe.
    if "vin" in df.columns:
        return "vin"
    for c in VIN_SYNONYMS:
        if c in df.columns:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Carpeta con los .xlsx diarios")
    ap.add_argument(
        "--out",
        required=True,
        help="Ruta de salida base; si no termina en .parquet se forzará a .parquet",
    )
    ap.add_argument("--normalizador", default=None, help="Ruta a normalizacionv2.py (opcional)")
    ap.add_argument("--whitelist", default=None, help="Ruta a whitelist.xlsx (opcional)")
    ap.add_argument(
        "--pattern",
        default="*_completo.xlsx",
        help="Patrón glob para seleccionar ficheros (por defecto '*_completo.xlsx')",
    )
    args = ap.parse_args()

    indir = Path(args.indir)
    if not indir.exists():
        raise SystemExit(f"⛔ No existe el directorio: {indir}")

    # Selecciona ficheros según patrón
    files = sorted(indir.glob(args.pattern))
    if not files:
        # Como fallback suave, si el patrón era *_completo.xlsx, intenta los base
        if args.pattern == "*_completo.xlsx":
            files = sorted(indir.glob("fichas_vehiculos_*.xlsx"))
        if not files:
            raise SystemExit(f"⛔ No se encontraron .xlsx en {indir} con patrón '{args.pattern}'")

    rows = []
    for p in files:
        # Lee Excel
        df = pd.read_excel(p, engine="openpyxl")

        # Anota trazabilidad
        df["__file"] = p.name
        df["__file_date"] = parse_file_date(p)

        # Normaliza VIN
        vin_col = choose_vin_column(df)
        if vin_col is None:
            raise SystemExit(
                f"⛔ No se encontró columna VIN en {p.name} (esperado 'vin' o sinónimos: {', '.join(VIN_SYNONYMS)})"
            )
        if "vin" not in df.columns:
            df["vin"] = df[vin_col]

        # Limpieza del VIN
        df["vin"] = df["vin"].astype(str).str.strip().str.upper()

        rows.append(df)

    merged = pd.concat(rows, ignore_index=True)

    # Contador de repeticiones de VIN dentro del mes
    counts = merged.groupby("vin", dropna=False).size().rename("vin_repeat").reset_index()
    merged = merged.merge(counts, on="vin", how="left")

    # Mantener la última ocurrencia por VIN según fecha de archivo
    merged = (
        merged.sort_values(["vin", "__file_date"])
        .groupby("vin", as_index=False, dropna=False)
        .tail(1)
    )

    # Limpia columnas auxiliares antes de guardar
    merged = merged.drop(columns=["__file", "__file_date"], errors="ignore")

    out_path = Path(args.out)
    out_parquet = out_path if out_path.suffix.lower() == ".parquet" else out_path.with_suffix(".parquet")

    # --- Robustness: coerce object-like columns to text to avoid pyarrow errors ---
    # (QUIRÚRGICO: añadido justo antes de escribir el parquet; no cambia la lógica)
    import json

    def _coerce_obj_cell(v):
        # Bytes -> str (utf-8 fallback latin1)
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode('utf-8')
            except Exception:
                return v.decode('latin-1', errors='ignore')
        # dict/list -> json string
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        # None/NaN -> empty string
        if pd.isna(v):
            return ""
        return str(v)

    for col in merged.select_dtypes(include=['object']).columns:
        types = merged[col].dropna().map(type).value_counts()
        if len(types) > 1:
            print(f"⚠️ Column '{col}' has mixed types: {types.to_dict()}; coercing to text.")
        merged[col] = merged[col].map(_coerce_obj_cell)
    # --- fin robustness ---

    # ✅ Guardar en Parquet (formato oficial)
    merged.to_parquet(out_parquet, index=False)
    print(f"✅ Escrito merged (con vin_repeat) → {out_parquet}")

    # 🔎 Opcional: Excel de inspección manual (descomentar si lo necesitas)
    # merged.to_excel(out_path.with_suffix(".xlsx"), index=False)

    # Normalizador: si todavía lo usas, invócalo sobre el mismo parquet (entrada = salida)
    if args.normalizador:
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parent.parent
        weights_file = repo_root / "weights.json"
        if not weights_file.exists():
            print(f"⚠️ Aviso: no existe {weights_file}. El normalizador usará los pesos por defecto internos.")

        cmd = [
            sys.executable, args.normalizador, str(out_parquet),
            "--out", str(out_parquet),
            "--weights", str(weights_file),
        ]
        if args.whitelist:
            cmd += ["--whitelist", args.whitelist]
        print("⚙️  Ejecutando normalizador:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(repo_root))
        print(f"✅ Normalización completada → {out_parquet}")

if __name__ == "__main__":
    main()
