#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Descarga todos los ficheros del tipo:
    fichas_vehiculos_YYYYMMDD_completo(.xlsx|Google Sheet)
para el mes indicado (argumento --year-month), ignorando los metadatos
de creación/modificación en Drive y filtrando por **nombre**.

- Si el archivo es Google Sheet, lo exporta a .xlsx.
- Guarda todo en --outdir (por defecto: monthly_inputs/).

Requisitos:
- GOOGLE_OAUTH_B64 y alcance Drive ya configurados (como usas en el resto del repo).
- Módulo helper `gdrive_auth.authenticate_drive()` disponible en tu proyecto.

Uso:
    python scripts/drive_fetch_month.py \
        --folder "$BCA_RESULTS_FOLDER_ID" \
        --outdir monthly_inputs \
        --year-month 2025-09
"""

import argparse
import datetime as dt
import io
import re
import sys
from pathlib import Path

from googleapiclient.http import MediaIoBaseDownload

# Tu helper de autenticación. Mantén el import como en tu repo.
from gdrive_auth import authenticate_drive

# ---------------------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------------------

# Coincide con cualquier YYYYMMDD dentro del nombre (2000..2099)
DATE_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
GSHEET_MIME = "application/vnd.google-apps.spreadsheet"


def _ym_tuple(ym: str) -> tuple[int, int]:
    """Devuelve (YYYY, MM) a partir de 'YYYY-MM'."""
    try:
        y, m = map(int, ym.split("-"))
        assert 1 <= m <= 12
        return y, m
    except Exception as e:
        raise ValueError(
            f"--year-month debe ser 'YYYY-MM'. Valor recibido: {ym}"
        ) from e


def _name_belongs_to_month(name: str, ym: str, require_completo: bool = True) -> bool:
    """
    Devuelve True si:
      - el nombre contiene un patrón YYYYMMDD
      - (opcional) contiene '_completo'
      - y su YYYY-MM coincide con `ym`
    """
    if require_completo and "_completo" not in name:
        return False
    m = DATE_RE.search(name)
    if not m:
        return False
    y, mo, _ = map(int, m.groups())
    Y, M = _ym_tuple(ym)
    return y == Y and mo == M


def _download_file(service, file_id: str, target: Path, mime_type: str) -> None:
    """
    Descarga/exporta un fichero de Drive a `target`.
    - Si es Google Sheet, exporta a XLSX.
    - Si ya es XLSX, descarga el binario tal cual.
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    if mime_type == GSHEET_MIME:
        request = service.files().export_media(fileId=file_id, mimeType=XLSX_MIME)
    else:
        request = service.files().get_media(fileId=file_id)

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    with open(target, "wb") as f:
        f.write(buf.getvalue())


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch mensual por nombre de fichero.")
    p.add_argument("--folder", required=True, help="Folder ID de Drive (BCA_RESULTS_FOLDER_ID, etc.)")
    p.add_argument("--outdir", default="monthly_inputs", help="Directorio local destino (por defecto: monthly_inputs)")
    p.add_argument("--year-month", required=True, help="Mes objetivo en formato YYYY-MM (p.ej. 2025-09)")
    p.add_argument(
        "--allow-base",
        action="store_true",
        help="Permitir también fichas_vehiculos_YYYYMMDD (sin _completo). Por defecto, SOLO *_completo.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    ym = args["year_month"] if isinstance(args, dict) else args.year_month  # por si se llama desde otros runners
    require_completo = not args.allow_base

    # Autenticación (usa GOOGLE_OAUTH_B64 y scope definidos en tu CI)
    service = authenticate_drive()

    # Query por parent + tipo + prefijo de nombre. El filtrado fino (YYYYMMDD/_completo/mes) lo hace Python.
    q = " and ".join([
        f"'{args.folder}' in parents",
        "trashed = false",
        "("
        f" mimeType = '{XLSX_MIME}'"
        f" or mimeType = '{GSHEET_MIME}'"
        ")",
        "name contains 'fichas_vehiculos_'",
    ])

    page_token = None
    total = 0
    print(f"🔎 Listing files for month {ym} (by filename pattern: fichas_vehiculos_YYYYMMDD{'_completo' if require_completo else ''})")

    while True:
        resp = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
            spaces="drive",
            orderBy="name",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        for f in resp.get("files", []):
            fid = f["id"]
            name = f["name"]
            mime = f.get("mimeType", "")

            # Filtrado por nombre -> YYYYMMDD + (opcional) _completo + mes=ym
            if not _name_belongs_to_month(name, ym, require_completo=require_completo):
                continue

            # Nombre de salida .xlsx
            out_name = name if name.lower().endswith(".xlsx") else (name + ".xlsx")
            target = outdir / out_name

            _download_file(service, fid, target, mime)
            total += 1
            print(f"⬇️  Downloaded: {target.name}")

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    print(f"✅ Total files downloaded for {ym}: {total}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.", file=sys.stderr)
        raise
