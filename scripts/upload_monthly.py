#!/usr/bin/env python3
import argparse, os, sys
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from gdrive_auth import authenticate_drive  # requiere GOOGLE_OAUTH_B64

def _find_in_folder(service, folder_id: str, name: str):
    q = " and ".join([
        f"'{folder_id}' in parents",
        "trashed = false",
        f"name = '{name}'",
    ])
    r = service.files().list(q=q, fields="files(id,name)", pageSize=1).execute()
    files = r.get("files", [])
    return files[0] if files else None

def upload_or_update(service, folder_id: str, path: str):
    if not os.path.exists(path):
        print(f"⚠️ No existe {path}, salto.", file=sys.stderr)
        return

    name = os.path.basename(path)
    # Sugerimos mimetype de .xlsx (no es obligatorio)
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if name.lower().endswith(".xlsx") else None
    media = MediaFileUpload(path, mimetype=mime, resumable=True)

    try:
        existing = _find_in_folder(service, folder_id, name)
        if existing:
            service.files().update(fileId=existing["id"], media_body=media).execute()
            print(f"🔁 Actualizado {name}")
        else:
            body = {"name": name, "parents": [folder_id]}
            service.files().create(body=body, media_body=media, fields="id").execute()
            print(f"✅ Subido {name}")
    except HttpError as e:
        print(f"❌ Error subiendo {name}: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder ID de destino en Drive")
    ap.add_argument("--files", required=True, help="Lista separada por comas")
    args = ap.parse_args()

    service = authenticate_drive()  # usa GOOGLE_OAUTH_B64
    for f in [p.strip() for p in args.files.split(",") if p.strip()]:
        upload_or_update(service, args.folder, f)
