#!/usr/bin/env python3
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gdrive_auth import authenticate_drive  # ya existe en tu repo
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

PARQUET_PATH = os.getenv("GANVAM_PARQUET", "ganvam_fase2_normalizado.parquet")
FOLDER_ID = os.getenv("GANVAM_PARQUET_FOLDER_ID", "").strip()

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
    # MIME de Parquet (válido); si diera problemas, usar None o application/octet-stream
    mime = "application/vnd.apache.parquet"
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

def main():
    folder_id = ensure_env("GANVAM_PARQUET_FOLDER_ID")
    service = authenticate_drive()  # usa tu flujo OAuth/Service Account existente
    target = PARQUET_PATH
    if not os.path.exists(target):
        print(f"⛔ No existe el Parquet esperado: {target}", file=sys.stderr)
        sys.exit(1)
    upload_or_update(service, folder_id, target)

if __name__ == "__main__":
    main()
