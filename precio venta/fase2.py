#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 2 – Ganvam (Precio de Venta)
---------------------------------
Lee ganvam_esqueleto.json (Fase 1) y rasca los endpoints de vehículos para
cada (marca, modelo, combustible, año). Genera:
  - ganvam_fase2_resultados.csv               (crudo rascado)
  - ganvam_fase2_huecos.csv                   (años faltantes detectados)
  - ganvam_fase2_coverage_summary.csv         (resumen de cobertura)
  - ganvam_fase2_normalizado.parquet          (normalización vía normalizacionv2.py en la RAÍZ DEL REPO)

La llamada a normalizacionv2.py se mantiene EXACTAMENTE como en el repo
(ubicación en raíz del repo, con whitelist.xlsx también en raíz).
"""

import os
import sys
import json
import csv
import time
import traceback
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================
# CONFIG
# ============================
EMPRESA = int(os.getenv("GANVAM_EMPRESA", "60"))
PERIODO = int(os.getenv("GANVAM_PERIODO", "212"))  # Debe coincidir con Fase 1
BASE    = os.getenv("GANVAM_BASE", "https://webapi.ganvam.es/api/AutomocionTarifaVOVentas")

SLEEP   = float(os.getenv("GANVAM_SLEEP", "0.30"))
RETRIES = int(os.getenv("GANVAM_RETRIES", "3"))
PAUSE_IN_WORKER = 0.25  # pausa ligera entre llamadas de un mismo worker

# Rutas y archivos (relativos a este script)
BASE_DIR     = Path(__file__).resolve().parent             # carpeta "precio venta"
ESQUELETO_JSON = BASE_DIR / "ganvam_esqueleto.json"
RESULTS_CSV     = BASE_DIR / "ganvam_fase2_resultados.csv"
FAILED_CSV      = BASE_DIR / "ganvam_fase2_fallos.csv"
HUECOS_CSV      = BASE_DIR / "ganvam_fase2_huecos.csv"
COVERAGE_CSV    = BASE_DIR / "ganvam_fase2_coverage_summary.csv"
PARQUET_OUT     = BASE_DIR / "ganvam_fase2_normalizado.parquet"

# ============================
# AUTH + SESSION
# ============================
def ensure_token() -> str:
    tok = os.getenv("GANVAM_TOKEN", "").strip()
    if tok:
        print("[auth] Usando GANVAM_TOKEN desde el entorno.")
        return tok
    if not sys.stdin.isatty():
        print("⛔ Falta GANVAM_TOKEN (entorno no interactivo).")
        sys.exit(1)
    manual = input("Pega tu JWT (Bearer <TOKEN>): ").strip()
    if not manual:
        print("No hay token. Saliendo.")
        sys.exit(1)
    return manual

GANVAM_TOKEN = ensure_token()

SESSION = requests.Session()
adapter = HTTPAdapter(
    pool_connections=32,
    pool_maxsize=32,
    max_retries=Retry(
        total=RETRIES,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    ),
)
SESSION.mount("https://", adapter)
SESSION.headers.update({
    "Authorization": f"Bearer {GANVAM_TOKEN}",
    "User-Agent": "BCA-Ganvam-Fase2/1.1",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
})

def get_json(url: str):
    for i in range(RETRIES):
        try:
            r = SESSION.get(url, timeout=45)
            if r.status_code == 200:
                time.sleep(SLEEP)
                return r.json()
            if r.status_code in (204, 404):
                time.sleep(SLEEP)
                return []
            time.sleep(0.25 + 0.2 * i)
        except Exception:
            time.sleep(0.25 + 0.2 * i)
    return None

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

# ============================
# CARGA ESQUELETO
# ============================
def cargar_esqueleto():
    if not ESQUELETO_JSON.exists():
        print(f"⛔ No existe {ESQUELETO_JSON}. ¿Ejecutaste Fase 1?")
        sys.exit(1)
    with open(ESQUELETO_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ============================
# SCRAPING
# ============================
def scrape_endpoint(marca, modelo, comb_id, comb_desc, anio, endpoint_vehiculos):
    """Rasca un endpoint concreto y devuelve (rows, failed?)."""
    data = get_json(endpoint_vehiculos)
    rows = []
    failed = False
    if data is None:
        failed = True
        return rows, failed
    if not isinstance(data, list):
        failed = True
        return rows, failed

    for v in data:
        row = dict(v)  # copia campos originales
        row["marca_raw"] = marca
        row["modelo_raw"] = modelo
        row["combustible_id"] = comb_id
        row["combustible_desc"] = comb_desc
        row["anio"] = to_int(anio)
        # Asegurar numéricos comunes
        row["startYear"] = to_int(row.get("startYear") or row.get("matriculacion"))
        row["endYear"]   = to_int(row.get("endYear")   or row.get("matriculacion"))
        rows.append(row)

    return rows, failed

def run_scraping(esqueleto):
    all_rows = []
    failed_endpoints = []

    trabajos = []
    for m in esqueleto.get("marcas", []):
        marca = m.get("marca")
        for mod in m.get("modelos", []):
            modelo = mod.get("modelo")
            for cb in mod.get("combustibles", []):
                comb_id   = cb.get("combustible_id")
                comb_desc = cb.get("combustible_desc")
                for a in cb.get("anios", []):
                    anio = a.get("anio")
                    endpoint = a.get("endpoint_vehiculos")
                    if not endpoint or anio is None:
                        continue
                    trabajos.append((marca, modelo, comb_id, comb_desc, anio, endpoint))

    print(f"🧵 Trabajos a ejecutar: {len(trabajos)}")
    with ThreadPoolExecutor(max_workers=int(os.getenv("GANVAM_W_RASCADORES", "16"))) as ex:
        futures = [ex.submit(scrape_endpoint, *t) for t in trabajos]
        for t, fut in zip(trabajos, as_completed(futures)):
            try:
                rows, failed = fut.result()
                all_rows.extend(rows)
                if failed:
                    failed_endpoints.append(t[-1])
            except Exception:
                failed_endpoints.append(t[-1])
                traceback.print_exc()
            finally:
                time.sleep(PAUSE_IN_WORKER)

    print(f"✅ Filas OK: {len(all_rows)}")
    print(f"⚠️ Endpoints fallidos definitivos: {len(failed_endpoints)}")

    # Guardar fallidos si existen
    if failed_endpoints:
        with open(FAILED_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["endpoint"])
            for u in failed_endpoints:
                w.writerow([u])
        print(f"💾 Fallidos guardados: {FAILED_CSV}")

    return all_rows, failed_endpoints

# ============================
# GUARD DE SANIDAD + COBERTURA (NUEVO)
# ============================
def generar_reportes_cobertura(all_rows, esqueleto):
    import collections

    # 1) Años esperados según esqueleto
    expected = collections.defaultdict(set)   # (marca,modelo,comb_desc) -> set(anios)
    for m in esqueleto.get("marcas", []):
        mk = m.get("marca")
        for mod in m.get("modelos", []):
            md = mod.get("modelo")
            for cb in mod.get("combustibles", []):
                cd = cb.get("combustible_desc")
                for a in cb.get("anios", []):
                    y = a.get("anio")
                    if mk and md and cd and isinstance(y, int):
                        expected[(mk, md, cd)].add(y)

    # 2) Años presentes en resultados + min startYear observada
    present = collections.defaultdict(set)
    min_start_seen = {}
    for r in all_rows:
        mk = r.get("marca_raw")
        md = r.get("modelo_raw")
        cd = r.get("combustible_desc")
        y  = r.get("anio")
        if mk and md and cd and isinstance(y, int):
            key = (mk, md, cd)
            present[key].add(y)
            s = r.get("startYear")
            if isinstance(s, int):
                min_start_seen[key] = s if key not in min_start_seen else min(min_start_seen[key], s)

    # 3) Huecos vs esqueleto
    huecos = []
    for key, exp_years in expected.items():
        miss = sorted(exp_years - present.get(key, set()))
        for y in miss:
            huecos.append([*key, y, "faltante_vs_esqueleto"])

    # 4) Historia perdida por startYear
    for key, smin in min_start_seen.items():
        if present.get(key):
            min_present = min(present[key])
            if isinstance(smin, int) and smin < min_present:
                for y in range(smin, min_present):
                    huecos.append([*key, y, "faltante_por_startYear"])

    # 5) Volcar reportes
    if huecos:
        with open(HUECOS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["marca_raw","modelo_raw","combustible_desc","anio_faltante","motivo"])
            w.writerows(huecos)
        print(f"⚠️ Huecos detectados ({len(huecos)}). Ver: {HUECOS_CSV}")

    with open(COVERAGE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["marca_raw","modelo_raw","combustible_desc","anios_esperados","anios_presentes","faltar_n"])
        for key in sorted(set(expected.keys()) | set(present.keys())):
            exp = sorted(expected.get(key, set()))
            prs = sorted(present.get(key, set()))
            miss_n = len(set(exp) - set(prs)) if exp else 0
            w.writerow([*key, ";".join(map(str,exp)) or "-", ";".join(map(str,prs)) or "-", miss_n])
    print(f"🧾 Coverage resumen → {COVERAGE_CSV}")

# ============================
# IO RESULTADOS
# ============================
def volcar_csv(all_rows):
    if not all_rows:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["marca_raw","modelo_raw","combustible_id","combustible_desc","anio"])
        print(f"📝 CSV vacío → {RESULTS_CSV}")
        return

    # Cabecera unificada
    keys = set()
    for r in all_rows:
        keys.update(r.keys())
    keys = list(sorted(keys))

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"💾 Resultados CSV → {RESULTS_CSV} ({len(all_rows)} filas)")

# ============================
# MAIN
# ============================
def main():
    print(f"➡️  Fase 2 | Empresa={EMPRESA} Periodo={PERIODO} | BASE={BASE}")
    esqueleto = cargar_esqueleto()

    # Scraping
    all_rows, failed_endpoints = run_scraping(esqueleto)

    # ----- NUEVO: Guard de sanidad + cobertura -----
    generar_reportes_cobertura(all_rows, esqueleto)
    # ----- FIN NUEVO -----

    # CSV crudo
    volcar_csv(all_rows)

    # Normalización (CLI) -> Parquet
    try:
        print("🧹 Normalizando con whitelist...")
        # EXACTAMENTE COMO EN TU REPO: normalizador y whitelist en la RAÍZ
        repo_root   = Path(__file__).resolve().parent.parent  # sube un nivel desde "precio venta/"
        normalizador = repo_root / "normalizacionv2.py"
        whitelist    = repo_root / "whitelist.xlsx"

        subprocess.run([
            sys.executable, str(normalizador),
            "--input", str(RESULTS_CSV),
            "--whitelist", str(whitelist),
            "--out", str(PARQUET_OUT),
            "--out-col", "modelo_base"
        ], check=True)

        print(f"✅ Normalización terminada → {PARQUET_OUT}")
    except Exception as e:
        print("⚠️ Error ejecutando normalización vía CLI:", e)

if __name__ == "__main__":
    main()
