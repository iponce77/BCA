# scripts/dgt/automatizacion_dgt.py
from __future__ import annotations
import os, io, re, zipfile, tempfile, json
from pathlib import Path
from typing import List, Dict, Set
import requests
from bs4 import BeautifulSoup
import pandas as pd
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

# 1) Autenticación Drive (reutiliza helper del repo)
from gdrive_auth import authenticate_drive  # usa GOOGLE_OAUTH_B64 + scope drive.file
# 2) Parser -> Parquet
from scripts.DGT.parse_dgt import txt_to_parquet

URL_DGT = "https://www.dgt.es/menusecundario/dgt-en-cifras/matraba-listados/transacciones-automoviles-mensual.html"  # :contentReference[oaicite:9]{index=9}

# -------- Utils Drive --------
def get_drive():
    return authenticate_drive()

def list_parquets_in_folder(service, folder_id: str) -> Dict[str, str]:
    """Devuelve {yyyymm: fileId} para los .parquet en la carpeta."""
    q = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    files = []
    page_token = None
    while True:
        resp = service.files().list(q=q, fields="nextPageToken, files(id, name, mimeType)", pageToken=page_token).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    meses: Dict[str, str] = {}
    for f in files:
        name = f["name"]
        if name.lower().endswith(".parquet"):
            m = re.search(r"(20\d{2})(\d{2})", name)
            if m:
                yyyymm = f"{m.group(1)}{m.group(2)}"
                meses[yyyymm] = f["id"]
    return meses

def upload_parquet(service, folder_id: str, local_path: Path, target_name: str) -> str:
    file_metadata = {"name": target_name, "parents": [folder_id]}
    media = MediaFileUpload(str(local_path), mimetype="application/octet-stream", resumable=True)
    created = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
    return created["id"]

# -------- DGT scraping --------
def list_dgt_zip_urls() -> Dict[str, str]:
    """Devuelve {yyyymm: url_zip} desde la página de DGT."""
    r = requests.get(URL_DGT, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    urls = [a["href"] for a in soup.find_all("a", href=True) if a["href"].lower().endswith(".zip")]
    out: Dict[str, str] = {}
    for u in urls:
        name = u.split("/")[-1]
        m = re.search(r"(20\d{2})(\d{2})", name)
        if m:
            yyyymm = f"{m.group(1)}{m.group(2)}"
            out[yyyymm] = u
    return out

# -------- Normalización (normalizacionv2) --------
def run_normalizacionv2_over_parquet(parquet_path: Path, normalizador_path: Path, whitelist_path: Path) -> None:
    """
    Bridge: parquet -> tmp csv -> normalizacionv2 (xlsx) -> merge columnas -> parquet (overwrite).
    No deja residuos; no conserva CSV ni XLSX.
    """
    import subprocess, tempfile
    df_orig = pd.read_parquet(parquet_path)

    with tempfile.TemporaryDirectory() as td:
        tmp_csv = Path(td) / (parquet_path.stem + ".csv")
        out_xlsx = Path(td) / (parquet_path.stem + "_norm.xlsx")
        df_orig.to_csv(tmp_csv, index=False)

        cmd = [
            "python", str(normalizador_path),
            "--input", str(tmp_csv),
            "--whitelist", str(whitelist_path),
            "--make-col", "marca",
            "--model-col", "submodelo",
            "--out", str(out_xlsx),
        ]
        weights = os.environ.get("NORMALIZADOR_WEIGHTS_PATH")
        if weights:
            cmd += ["--weights", str(weights)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not out_xlsx.exists():
            raise RuntimeError(f"normalizacionv2 falló.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

        df_norm = pd.read_excel(out_xlsx)
        # columnas típicas a inyectar (ajústalo si tu normalizador añade otras)
        for col in ["modelo_base", "make_clean", "modelo_detectado", "MARCA", "MODELO"]:
            if col in df_norm.columns:
                df_orig[col] = df_norm[col]

    df_orig.to_parquet(parquet_path, index=False)

# -------- Main pipeline --------
def main():
    folder_id = os.environ["DGT_PARQUET_FOLDER_ID"]  # secret
    normalizador = os.environ.get("NORMALIZADOR_DGT_PATH", "normalizacionv2.py")
    whitelist = os.environ.get("NORMALIZADOR_WHITELIST_PATH", "whitelist.xlsx")

    if not Path(normalizador).exists():
        raise FileNotFoundError(f"No encuentro normalizador: {normalizador}")
    if not Path(whitelist).exists():
        raise FileNotFoundError(f"No encuentro whitelist: {whitelist}")

    service = get_drive()

    # 1) Inventario Drive
    ya_subidos = list_parquets_in_folder(service, folder_id)   # {yyyymm: fileId}

    # 2) Catálogo DGT online
    disponibles = list_dgt_zip_urls()                          # {yyyymm: url_zip}

    # 3) Determinar pendientes
    pendientes = sorted([ym for ym in disponibles.keys() if ym not in ya_subidos])

    if not pendientes:
        print("No hay meses nuevos.")
        return

    # 4) Descarga + extracción + parse + normalización + upload
    workdir = Path("dgt_work")
    workdir.mkdir(exist_ok=True)

    for yyyymm in pendientes:
        url = disponibles[yyyymm]
        zip_name = f"dgt_{yyyymm}.zip"
        zip_path = workdir / zip_name

        print(f"Descargando {zip_name} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk: f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(workdir)

        # Encontrar .txt (puede haber subcarpetas)
        txts = [p for p in workdir.rglob("*.txt")]
        if not txts:
            print(f"⚠️ No se encontraron TXT en {zip_name}")
            continue

        # Por cada .txt -> parquet (puedes también concatenar si DGT parte el mes)
        for txt in txts:
            out_parquet = workdir / (re.sub(r"\.txt$", ".parquet", txt.name, flags=re.I))
            txt_to_parquet(txt, out_parquet)  # Parquet base sin normalización

            # Normalización v2 sobre el Parquet
            run_normalizacionv2_over_parquet(out_parquet, Path(normalizador), Path(whitelist))

            # Nombre final consistente: dgt_transmisiones_YYYMM.parquet
            target_name = f"dgt_transmisiones_{yyyymm}.parquet"
            print(f"Subiendo {target_name} a Drive...")
            upload_parquet(service, folder_id, out_parquet, target_name)

        # Limpieza de residuos del zip actual
        for p in workdir.iterdir():
            if p.is_file() and p.suffix.lower() in {".zip", ".txt", ".parquet"}:
                p.unlink()
            elif p.is_dir():
                # cuidado de no borrar workdir
                for q in p.rglob("*"):
                    if q.is_file(): q.unlink()
                try: p.rmdir()
                except OSError: pass

if __name__ == "__main__":
    main()
