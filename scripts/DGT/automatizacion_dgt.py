# scripts/DGT/automatizacion_dgt.py
from __future__ import annotations
import os
import re
import io
import time
import zipfile
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from gdrive_auth import authenticate_drive  # usa GOOGLE_OAUTH_B64 + scope drive.file
from scripts.DGT.parse_dgt import txt_to_parquet

# =========================
#   LOGGING + TIMERS
# =========================
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("automatizacion_dgt")

def tic():
    return time.perf_counter()

def lap(t0: float, msg: str):
    logger.info(f"{msg} | {time.perf_counter() - t0:.1f}s")


# =========================
#   CONSTANTES
# =========================
URL_DGT = (
    "https://www.dgt.es/menusecundario/dgt-en-cifras/matraba-listados/transacciones-automoviles-mensual.html"
)


# =========================
#   DRIVE HELPERS
# =========================
def get_drive():
    return authenticate_drive()

def list_parquets_in_folder(service, folder_id: str) -> Dict[str, str]:
    """Devuelve {yyyymm: fileId} para los .parquet en la carpeta."""
    q = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    meses: Dict[str, str] = {}
    page_token = None
    while True:
        resp = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()
        for f in resp.get("files", []):
            name = f["name"]
            if name.lower().endswith(".parquet"):
                m = re.search(r"(20\d{2})(\d{2})", name)
                if m:
                    yyyymm = f"{m.group(1)}{m.group(2)}"
                    meses[yyyymm] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return meses

def upload_parquet(service, folder_id: str, local_path: Path, target_name: str) -> str:
    """
    Subida robusta a Drive:
    1) Varios intentos con backoff exponencial.
    2) Re-crea MediaFileUpload en cada intento.
    3) Primero resumable con chunksize fijo; si falla varias veces, fallback a simple upload.
    """
    file_metadata = {"name": target_name, "parents": [folder_id]}

    # Intentos totales (resumable) antes de probar simple upload
    MAX_RESUMABLE_TRIES = 5
    # Intentos para simple upload si lo anterior falla
    MAX_SIMPLE_TRIES = 2
    # Chunk de ~8 MB (puedes bajar a 5 MB si la red es inestable)
    CHUNK_SIZE = 8 * 1024 * 1024

    last_err: Optional[Exception] = None

    # 1) Reintentos con upload RESUMABLE
    for attempt in range(1, MAX_RESUMABLE_TRIES + 1):
        try:
            media = MediaFileUpload(
                local_path.as_posix(),
                mimetype="application/octet-stream",
                resumable=True,
                chunksize=CHUNK_SIZE,
            )
            req = service.files().create(
                body=file_metadata, media_body=media, fields="id"
            )
            # num_retries aquí pide a googleapiclient reintentar internamente
            created = req.execute(num_retries=3)
            return created["id"]
        except HttpError as e:
            last_err = e
            wait = min(30, 2 ** attempt)  # backoff exponencial con tope
            # Log claro con intento/tipo de subida
            logger.warning(f"[Drive resumable] intento {attempt}/{MAX_RESUMABLE_TRIES} falló: {e}\nReintentando en {wait}s…")
            time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = min(30, 2 ** attempt)
            logger.warning(f"[Drive resumable] intento {attempt}/{MAX_RESUMABLE_TRIES} falló (no-HttpError): {e}\nReintentando en {wait}s…")
            time.sleep(wait)

    # 2) Fallback: SIMPLE UPLOAD (sin resumable), algunos entornos/proxys van mejor así
    for attempt in range(1, MAX_SIMPLE_TRIES + 1):
        try:
            media = MediaFileUpload(
                local_path.as_posix(),
                mimetype="application/octet-stream",
                resumable=False,
            )
            created = service.files().create(
                body=file_metadata, media_body=media, fields="id"
            ).execute(num_retries=3)
            logger.info("[Drive simple] subida OK (fallback).")
            return created["id"]
        except Exception as e:
            last_err = e
            wait = 3 * attempt
            logger.warning(f"[Drive simple] intento {attempt}/{MAX_SIMPLE_TRIES} falló: {e}\nReintentando en {wait}s…")
            time.sleep(wait)

    # Si llegamos aquí, fallaron todas las vías
    raise RuntimeError(f"Fallo subiendo {target_name} a Drive tras reintentos: {last_err}")


# =========================
#   DGT SCRAPING
# =========================
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


# =========================
#   NORMALIZACIÓN v2 (bridge)
# =========================
def run_normalizacionv2_over_parquet(
    parquet_path: Path,
    normalizador_path: Path,
    whitelist_path: Path,
    weights_env_var: str = "NORMALIZADOR_WEIGHTS_PATH",
    cols_minimas_para_normalizador: Optional[List[str]] = None,
) -> None:
    """
    Bridge parquet -> tmp CSV -> normalizacionv2 (XLSX) -> merge columnas -> parquet (overwrite).
    Exporta sólo las columnas que necesita el normalizador para ahorrar IO.
    """
    import subprocess

    if cols_minimas_para_normalizador is None:
        cols_minimas_para_normalizador = ["marca", "submodelo"]

    df_orig = pd.read_parquet(parquet_path)

    with tempfile.TemporaryDirectory() as td:
        tmp_csv = Path(td) / (parquet_path.stem + ".csv")
        out_xlsx = Path(td) / (parquet_path.stem + "_norm.xlsx")

        # Exporta sólo columnas necesarias; si faltan, crea vacías
        export_cols = [c for c in cols_minimas_para_normalizador if c in df_orig.columns]
        for c in cols_minimas_para_normalizador:
            if c not in export_cols:
                df_orig[c] = None
                export_cols.append(c)
        df_orig[export_cols].to_csv(tmp_csv, index=False)

        cmd = [
            "python", str(normalizador_path),
            "--input", str(tmp_csv),
            "--whitelist", str(whitelist_path),
            "--make-col", "marca",
            "--model-col", "submodelo",
            "--out", str(out_xlsx),
        ]
        weights = os.environ.get(weights_env_var)
        if weights:
            # Ajusta el flag si tu normalizador usa otro (--model, --weights-path, etc.)
            cmd += ["--weights", str(weights)]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not out_xlsx.exists():
            raise RuntimeError(
                "normalizacionv2 falló.\n"
                f"CMD: {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            )

        df_norm = pd.read_excel(out_xlsx)

        # Inyecta columnas típicas del normalizador si existen
        columnas_a_inyectar = [
            "modelo_base", "make_clean", "modelo_detectado", "MARCA", "MODELO"
        ]
        for col in columnas_a_inyectar:
            if col in df_norm.columns:
                df_orig[col] = df_norm[col]

    df_orig.to_parquet(parquet_path, index=False)


# =========================
#   MAIN PIPELINE
# =========================
def main():
    folder_id = os.environ["DGT_PARQUET_FOLDER_ID"]  # secret
    normalizador = Path(os.environ.get("NORMALIZADOR_DGT_PATH", "normalizacionv2.py"))
    whitelist = Path(os.environ.get("NORMALIZADOR_WHITELIST_PATH", "whitelist.xlsx"))

    if not normalizador.exists():
        raise FileNotFoundError(f"No encuentro normalizador: {normalizador}")
    if not whitelist.exists():
        raise FileNotFoundError(f"No encuentro whitelist: {whitelist}")

    t0 = tic()
    service = get_drive()
    lap(t0, "Autenticación Drive lista")

    t = tic()
    ya_subidos = list_parquets_in_folder(service, folder_id)
    lap(t, f"Inventario Drive OK ({len(ya_subidos)} meses)")

    t = tic()
    disponibles = list_dgt_zip_urls()
    lap(t, f"Listado DGT OK ({len(disponibles)} meses detectados)")

    pendientes = sorted([ym for ym in disponibles.keys() if ym not in ya_subidos])
    if not pendientes:
        logger.info("No hay meses nuevos.")
        return

    workdir = Path("dgt_work")
    workdir.mkdir(exist_ok=True)

    for yyyymm in pendientes:
        url = disponibles[yyyymm]
        zip_name = f"dgt_{yyyymm}.zip"
        zip_path = workdir / zip_name

        t = tic()
        logger.info(f"Descargando {zip_name} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        lap(t, f"Descarga ZIP {zip_name} completada")

        t = tic()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(workdir)
        lap(t, f"Extracción ZIP {zip_name} completada")

        # Buscar TXT en subcarpetas
        txts = [p for p in workdir.rglob("*.txt")]
        if not txts:
            logger.warning(f"⚠️ No se encontraron TXT en {zip_name}")
            continue

        for txt in txts:
            # Nombre final consistente por mes (si hay varios txt, puedes distinguirlos si quieres)
            target_name = f"dgt_transmisiones_{yyyymm}.parquet"
            out_parquet = workdir / (re.sub(r"\.txt$", ".parquet", txt.name, flags=re.I))

            # 1) TXT -> Parquet (streaming + PyArrow)
            t = tic()
            txt_to_parquet(txt, out_parquet, chunk_rows=50_000)
            lap(t, f"TXT→Parquet ({txt.name})")

            # 2) Normalizacion v2 (bridge CSV->XLSX->merge)
            t = tic()
            run_normalizacionv2_over_parquet(out_parquet, normalizador, whitelist)
            lap(t, f"normalizacionv2 ({txt.name})")

            # 3) Subir a Drive
            t = tic()
            logger.info(f"Subiendo {target_name} a Drive...")
            upload_parquet(service, folder_id, out_parquet, target_name)
            lap(t, f"Upload a Drive ({target_name})")

        # Limpieza de residuos del ZIP actual (opcional)
        for p in workdir.iterdir():
            if p.is_file() and p.suffix.lower() in {".zip", ".txt", ".parquet"}:
                try:
                    p.unlink()
                except OSError:
                    pass
            elif p.is_dir():
                for q in p.rglob("*"):
                    if q.is_file():
                        try:
                            q.unlink()
                        except OSError:
                            pass
                try:
                    p.rmdir()
                except OSError:
                    pass


if __name__ == "__main__":
    main()
