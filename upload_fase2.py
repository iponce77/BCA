#!/usr/bin/env python3
import os
import sys
import re
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

def trash_base_files(service, folder_id: str, completo_name: str):
    """
    Busca y mueve a la papelera los ficheros 'base' (mismo prefijo fecha) que NO contienen '_completo'.
    """
    m = re.search(r"(fichas_vehiculos_[0-9]{8})", completo_name)
    if not m:
        print("⚠️ No se detectó patrón 'fichas_vehiculos_YYYYMMDD' en el nombre; no se borrará nada.")
        return
    date_prefix = m.group(1)

    # construimos query: en la carpeta, no en papeleras, que contengan la fecha y NO contengan "_completo"
    q = " and ".join([
        f"'{folder_id}' in parents",
        "trashed = false",
        f"name contains '{date_prefix}'",
        "not name contains '_completo'"
    ])

    resp = service.files().list(q=q, fields="files(id,name)", pageSize=1000,
                               includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    files = resp.get("files", [])
    if not files:
        print("ℹ️ No se encontraron ficheros base a mover a la papelera.")
        return

    for f in files:
        fid = f["id"]
        fname = f["name"]
        try:
            # Preferimos 'mover a la papelera' en vez de borrar permanentemente
            service.files().update(fileId=fid, body={"trashed": True}).execute()
            print(f"🗑️ Movido a la papelera en Drive: {fname}")
        except HttpError as e:
            print(f"❌ Error moviendo a papelera {fname}: {e}", file=sys.stderr)

def pick_completo() -> str | None:
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

    # ---- NUEVO: tras subir correctamente, mover a la papelera los ficheros de fase 1 (si procede) ----
    # Si VAR de entorno KEEP_BASE_FILES es '1'/'true' no borramos (por seguridad)
    keep = os.environ.get("KEEP_BASE_FILES", "").lower()
    if keep in ("1", "true", "yes"):
        print("ℹ️ KEEP_BASE_FILES activado: no se moverán a la papelera los ficheros base.")
        return

    if "_completo" in os.path.basename(target):
        try:
            trash_base_files(service, folder_id, os.path.basename(target))
        except Exception as e:
            print(f"⚠️ Error durante el borrado de ficheros base: {e}", file=sys.stderr)
    else:
        print("ℹ️ El fichero subido no contiene '_completo' en su nombre; no se borrará nada por seguridad.")

if __name__ == "__main__":
    main()


