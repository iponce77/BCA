#!/usr/bin/env python3
import os
from glob import glob
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from gdrive_auth import authenticate_drive

def ensure_env(var: str) -> str:
    v = os.environ.get(var, "").strip()
    if not v:
        raise RuntimeError(f"Falta {var} en el entorno")
    return v

def find_in_folder(service, folder_id: str, name: str):
    q = " and ".join([
        f"'{folder_id}' in parents",
        "trashed = false",
        f"name = '{name}'",
    ])
    resp = service.files().list(q=q, fields="files(id,name)", pageSize=1).execute()
    files = resp.get("files", [])
    return files[0] if files else None

def upload_or_update(service, folder_id: str, file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    name = os.path.basename(file_path)
    # Mimetype sugerido para .xlsx (opcional, pero ayuda a Drive)
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if name.lower().endswith(".xlsx") else None
    media = MediaFileUpload(file_path, mimetype=mime, resumable=True)

    try:
        existing = find_in_folder(service, folder_id, name)
        if existing:
            service.files().update(fileId=existing["id"], media_body=media).execute()
            print(f"🔁 Actualizado en Drive: {name}")
        else:
            body = {"name": name, "parents": [folder_id]}
            service.files().create(body=body, media_body=media, fields="id").execute()
            print(f"⬆️ Subido a Drive: {name}")
    except HttpError as e:
        print(f"❌ Error subiendo {name}: {e}")
        raise

def pick_base() -> str | None:
    """
    Selecciona el archivo base a subir:
      1) Si EXCEL_IN (env) apunta a un fichero existente, usarlo.
      2) Buscar fichas_vehiculos_*.xlsx en el directorio actual.
      3) Buscar fichas_vehiculos_*.xlsx en out/.
      (Nunca archivos que terminen en _completo.xlsx)
    """
    excel_in = os.environ.get("EXCEL_IN", "").strip()
    if excel_in and os.path.exists(excel_in):
        return excel_in

    bases = sorted([p for p in glob("fichas_vehiculos_*.xlsx") if not p.endswith("_completo.xlsx")], reverse=True)
    if bases:
        return bases[0]

    bases_out = sorted([p for p in glob("out/fichas_vehiculos_*.xlsx") if not p.endswith("_completo.xlsx")], reverse=True)
    if bases_out:
        return bases_out[0]

    return None

def main():
    folder_id = ensure_env("GDRIVE_FOLDER_ID")
    service = authenticate_drive()
    target = pick_base()
    if not target:
        raise FileNotFoundError("No se encontró fichas_vehiculos_*.xlsx (base)")
    upload_or_update(service, folder_id, target)

if __name__ == "__main__":
    main()
