#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, sys, datetime as dt
from googleapiclient.http import MediaIoBaseDownload
from gdrive_auth import authenticate_drive

# ----------------------------
# Config
# ----------------------------
XLSX_MIME   = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
GSHEET_MIME = "application/vnd.google-apps.spreadsheet"


def resolve_target_date():
    """
    Devuelve YYYYMMDD (UTC). Si existe env TODAY=YYYYMMDD, lo usa para pruebas.
    """
    forced = os.environ.get("TODAY", "").strip()
    return forced if forced else dt.datetime.utcnow().strftime("%Y%m%d")


def _export_or_download(service, file_id: str, mime_type: str):
    """
    Si es Google Sheet, exporta a XLSX; si no, descarga binario.
    """
    if mime_type == GSHEET_MIME:
        return service.files().export_media(fileId=file_id, mimeType=XLSX_MIME)
    return service.files().get_media(fileId=file_id)


def find_by_name_today(service, folder_id: str, ymd: str):
    """
    Busca por nombre del día:
      - Prefijo 'fichas_vehiculos_'
      - Contiene YYYYMMDD
      - Prioriza *_completo si existe
    Acepta XLSX y Google Sheets. No usa metadatos de tiempo.
    """
    q = " and ".join([
        f"'{folder_id}' in parents",
        "trashed = false",
        "name contains 'fichas_vehiculos_'",
        f"name contains '{ymd}'",
        "("
        f" mimeType = '{XLSX_MIME}' or mimeType = '{GSHEET_MIME}'"
        ")"
    ])
    resp = service.files().list(
        q=q,
        fields="files(id,name,mimeType)",
        pageSize=1000,
        orderBy="name",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    if not files:
        return None
    completos = [f for f in files if "_completo" in f["name"]]
    return (completos or files)[0]


def find_latest_base(service, folder_id: str):
    """
    Fallback: si no hay del día, toma el último por nombre del prefijo.
    (Acepta también *_completo y Google Sheets.)
    """
    q = " and ".join([
        f"'{folder_id}' in parents",
        "trashed = false",
        "name contains 'fichas_vehiculos_'",
        "("
        f" mimeType = '{XLSX_MIME}' or mimeType = '{GSHEET_MIME}'"
        ")"
    ])
    resp = service.files().list(
        q=q,
        orderBy="name desc",
        fields="files(id,name,mimeType)",
        pageSize=1,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    files = resp.get("files", [])
    return files[0] if files else None


def download_file(service, file_id: str, local_name: str, mime_type: str):
    req = _export_or_download(service, file_id, mime_type)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    with open(local_name, "wb") as f:
        f.write(buf.getvalue())
    print(f"✅ Descargado: {local_name}")


def main():
    folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not folder_id:
        print("⛔ Falta GDRIVE_FOLDER_ID", file=sys.stderr); sys.exit(1)

    service = authenticate_drive()
    ymd = resolve_target_date()

    # 1) Busca el del día (prioriza *_completo)
    found = find_by_name_today(service, folder_id, ymd)

    # 2) Si no existe, toma el último del prefijo
    if not found:
        found = find_latest_base(service, folder_id)
        if not found:
            print("⛔ No hay base disponible en la carpeta de Drive.", file=sys.stderr)
            sys.exit(2)

    # Guarda siempre como .xlsx (si es Sheet, añadimos sufijo)
    out_name = found["name"] if found["name"].lower().endswith(".xlsx") else f"{found['name']}.xlsx"
    download_file(service, found["id"], out_name, found.get("mimeType", ""))


if __name__ == "__main__":
    main()
