#!/usr/bin/env python3
import os, sys
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
    # sugerimos mimetype para .xlsx (opcional)
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
        print(f"❌ Error subiendo {name}: {e}", file=sys.stderr)
        sys.exit(1)

def pick_completo() -> str | None:
    """
    Selecciona el archivo final a subir:
      1) Si EXCEL_FINAL (env) existe, usarlo.
      2) Buscar *_completo.xlsx en el directorio actual.
      3) Buscar out/*_completo.xlsx.
    """
    excel_final = os.environ.get("EXCEL_FINAL", "").strip()
    if excel_final and os.path.exists(excel_final):
        return excel_final

    cwd_matches = sorted(glob("*_completo.xlsx"), reverse=True)
    if cwd_matches:
        return cwd_matches[0]

    out_matches = sorted(glob("out/*_completo.xlsx"), reverse=True)
    if out_matches:
        return out_matches[0]

    return None

def main():
    folder_id = ensure_env("GDRIVE_FOLDER_ID")
    service = authenticate_drive()
    target = pick_completo()
    if not target:
        print("⛔ No se encontró ningún archivo *_completo.xlsx (ni en raíz ni en out/).", file=sys.stderr)
        sys.exit(1)
    upload_or_update(service, folder_id, target)

if __name__ == "__main__":
    main()
