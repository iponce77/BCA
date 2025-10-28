#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fase 1 – Ganvam (Precio de Venta)
---------------------------------
Construye el esqueleto jerárquico:
  idEmpresa, idPeriodo, base,
  marcas -> modelos -> combustibles -> años -> endpoint_vehiculos

Cambios:
- Añadir nodo de marca y de modelo **inmediatamente** (como en la versión estable),
  para no perder nodos si fallan sub-llamadas.
- Descubrir años completos usando startYear/endYear, rellenando desde el año de
  inicio hasta el final (limitado al año actual).
"""

import os, sys, json, time, urllib.parse, traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================
# CONFIG (cloud-ready)
# ============================
EMPRESA = int(os.getenv("GANVAM_EMPRESA", "60"))
PERIODO = int(os.getenv("GANVAM_PERIODO", "212"))   # si =0, se autodetecta (ver más abajo)
BASE    = os.getenv("GANVAM_BASE", "https://webapi.ganvam.es/api/AutomocionTarifaVOVentas")

SLEEP   = float(os.getenv("GANVAM_SLEEP", "0.35"))
RETRIES = int(os.getenv("GANVAM_RETRIES", "3"))
OUTFILE_JSON = os.getenv("GANVAM_FASE1_JSON", "ganvam_esqueleto.json")

MAX_WORKERS_MARCAS  = int(os.getenv("GANVAM_W_MARCAS", "8"))
MAX_WORKERS_MODELOS = int(os.getenv("GANVAM_W_MODELOS", "8"))

# ============================
# AUTH
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

TOKEN = ensure_token()

# Sesión HTTP con pool y reintentos
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
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/json",
    "User-Agent": "BCA-Ganvam-Fase1/1.1",
})

# ============================
# HELPERS
# ============================
q = lambda s: urllib.parse.quote(str(s), safe="")

def fetch_json(url: str, max_retries: int = None):
    """GET JSON con reintentos a nivel de requests + pequeño sleep entre llamadas."""
    tries = max_retries if max_retries is not None else RETRIES
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            # 404/204: sin datos → devuelve []
            if r.status_code in (204, 404):
                return []
            # otros: espera y reintenta
            time.sleep(0.3 + i * 0.2)
        except Exception:
            time.sleep(0.3 + i * 0.2)
    return None

def get_json(url: str):
    data = fetch_json(url)
    time.sleep(SLEEP)
    return data

def detecta_periodo_vigente(base: str, empresa: int, start: int, probe_n: int = 2) -> int:
    """Prueba start, start+1, ... y devuelve el primero con marcas no vacías."""
    for delta in range(0, probe_n + 1):
        cand = start + delta
        url = f"{base}/marcas/{empresa}/{cand}"
        data = get_json(url) or []
        if isinstance(data, list) and len(data) > 0:
            print(f"[periodo] Vigente detectado: {cand}")
            return cand
    print(f"[periodo] No se pudo detectar mejor; uso {start}")
    return start

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

# ============================
# API accessors
# ============================
def get_marcas(empresa: int, periodo: int):
    url = f"{BASE}/marcas/{empresa}/{periodo}"
    return get_json(url) or []

def get_modelos(empresa: int, periodo: int, marca: str):
    url = f"{BASE}/modelos/{empresa}/{periodo}/{q(marca)}"
    return get_json(url) or []

def get_combustibles(empresa: int, periodo: int, marca: str, modelo: str):
    url = f"{BASE}/combustibles/{empresa}/{periodo}/{q(marca)}/{q(modelo)}"
    return get_json(url) or []

def get_vehiculos(marca: str, modelo: str, comb_id: str, anio: int):
    """Devuelve (data, url) para ese año."""
    url = f"{BASE}/vehiculos/{EMPRESA}/{PERIODO}/{q(marca)}/{q(modelo)}/{q(comb_id)}/{anio}"
    data = get_json(url) or []
    return data, url

# ============================
# AÑOS (descubrimiento robusto)
# ============================
def inferir_anios_validos(marca: str, modelo: str, comb_id: str):
    """
    Heurística de sondeo: probamos varios años pivote y deducimos start/end
    a partir de 'vehiculos'. Así obtenemos el **rango completo** incluso si
    el sub-endpoint de años no lista históricos.
    """
    this_year = datetime.utcnow().year
    # Sondear una ventana deslizante de años recientes + anclas antiguas
    candidatos = [this_year, this_year-1, this_year-2, this_year-3, 2020, 2017, 2015, 2013]
    
    ejemplo_endpoint = None
    min_start, max_end = None, None

    for anio in candidatos:
        data, url = get_vehiculos(marca, modelo, comb_id, anio)
        if data and ejemplo_endpoint is None:
            ejemplo_endpoint = url  # primero que responde como ejemplo
        for row in (data or []):
            s = to_int(row.get("startYear") or row.get("matriculacion"))
            e = to_int(row.get("endYear")   or row.get("matriculacion"))
            if s is not None:
                min_start = s if (min_start is None or s < min_start) else min_start
            if e is not None:
                max_end = e if (max_end is None or e > max_end) else max_end

    anios = set()
    if min_start is not None:
        this_year = datetime.utcnow().year
        upper = min(max_end if max_end is not None else this_year, this_year)
        if upper < min_start:
            upper = min_start
        anios = set(range(min_start, upper + 1))
    else:
        # 🔁 Fallback: probar explícitamente los últimos 5 años por si el modelo es "nuevo"
        recientes = range(this_year, this_year-5, -1)
        min_start, max_end = None, None
        for y in recientes:
            data, _ = get_vehiculos(marca, modelo, comb_id, y)
            for v in (data or []):
                s = to_int(v.get("startYear") or v.get("matriculacion"))
                e = to_int(v.get("endYear")   or v.get("matriculacion"))
                if s is not None: min_start = s if (min_start is None or s < min_start) else min_start
                if e is not None: max_end = e if (max_end is None or e > max_end) else max_end
        if min_start is not None:
            upper = min(max_end if max_end is not None else this_year, this_year)
            if upper < min_start: upper = min_start
            anios = set(range(min_start, upper + 1))

    return sorted(anios), ejemplo_endpoint

def completar_rango_con_start_end(marca: str, modelo: str, comb_id: str, anios_base):
    """
    Refuerza años a partir de vehículos reales de los años base:
    consolida min(startYear) y max(endYear) y rellena el rango.
    """
    if not anios_base:
        return []

    min_start, max_end = None, None
    for y in anios_base:
        data, _ = get_vehiculos(marca, modelo, comb_id, y)
        for v in (data or []):
            s = to_int(v.get("startYear") or v.get("matriculacion"))
            e = to_int(v.get("endYear")   or v.get("matriculacion"))
            if s is not None:
                min_start = s if (min_start is None or s < min_start) else min_start
            if e is not None:
                max_end = e if (max_end is None or e > max_end) else max_end

    this_year = datetime.utcnow().year
    lower = min_start if min_start is not None else min(anios_base)
    upper = min(max_end if max_end is not None else this_year, this_year)
    if upper < lower:
        upper = lower

    return sorted(set(anios_base) | set(range(lower, upper + 1)))

# ============================
# BUILD ESQUELETO
# ============================
def construir_esqueleto():
    global PERIODO
    if PERIODO == 0:
        # autodetección opcional si pasas GANVAM_PERIODO=0
        PERIODO = detecta_periodo_vigente(BASE, EMPRESA, start=212, probe_n=3)

    arbol = {
        "idEmpresa": EMPRESA,
        "idPeriodo": PERIODO,
        "base": BASE,
        "marcas": []
    }

    # 1) Marcas
    marcas = get_marcas(EMPRESA, PERIODO) or []
    print(f"Marcas detectadas: {len(marcas)}")

    def procesa_marca(mk):
        try:
            # === claves como en el script antiguo (prioriza 'descripcion' / 'id') ===
            marca_name = mk.get("descripcion") or mk.get("id") or mk.get("marca") or mk
            marca_name = str(marca_name)

            nodo_marca = {
                "marca": marca_name,
                "endpoint_modelos": f"{BASE}/modelos/{EMPRESA}/{PERIODO}/{q(marca_name)}",
                "modelos": []
            }

            # ✅ Añadir **inmediatamente** como hacía la versión estable
            return_marca_ref = nodo_marca  # referencia para devolver
            # OJO: lo añadimos en el hilo principal (más abajo) para evitar carreras
            modelos = get_modelos(EMPRESA, PERIODO, marca_name) or []

            for md in modelos:
                modelo_name = md.get("id") or md.get("descripcion") or md.get("modelo") or md
                modelo_name = str(modelo_name)

                nodo_modelo = {
                    "modelo": modelo_name,
                    "endpoint_combustibles": f"{BASE}/combustibles/{EMPRESA}/{PERIODO}/{q(marca_name)}/{q(modelo_name)}",
                    "combustibles": []
                }

                # ✅ Añadir **inmediatamente** el modelo
                nodo_marca["modelos"].append(nodo_modelo)

                combustibles = get_combustibles(EMPRESA, PERIODO, marca_name, modelo_name) or []
                for cb in combustibles:
                    comb_id   = cb.get("id") or cb.get("combustible_id") or cb.get("Combustible") or cb.get("idCombustible")
                    comb_desc = cb.get("descripcion") or cb.get("combustible_desc") or cb.get("Descripcion") or str(comb_id)
                    if not comb_id:
                        continue

                    # A) Años inferidos por vehículos (sondeo)
                    anios, ejemplo_url = inferir_anios_validos(marca_name, modelo_name, str(comb_id))
                    # B) Refuerzo con backfill sobre base existente
                    anios = completar_rango_con_start_end(marca_name, modelo_name, str(comb_id), anios)

                    nodo_modelo["combustibles"].append({
                        "combustible_id": str(comb_id),
                        "combustible_desc": comb_desc,
                        "endpoint_ejemplo": ejemplo_url,
                        "anios": [
                            {
                                "anio": y,
                                "endpoint_vehiculos": f"{BASE}/vehiculos/{EMPRESA}/{PERIODO}/{q(marca_name)}/{q(modelo_name)}/{q(str(comb_id))}/{y}"
                            }
                            for y in anios
                        ]
                    })

            return return_marca_ref
        except Exception as e:
            print(f"[ERROR] procesando marca {mk}: {e}")
            traceback.print_exc()
            try:
                return {"marca": str(mk.get('descripcion') or mk.get('id') or mk), "modelos": []}
            except Exception:
                return None

    # Ejecutar en paralelo por marcas y **añadir inmediatamente** en el hilo principal
    resultados = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_MARCAS) as ex:
        futs = [ex.submit(procesa_marca, m) for m in marcas]
        for f in as_completed(futs):
            res = f.result()
            if isinstance(res, dict):
                resultados.append(res)

    # Ordenar por nombre de marca para estabilidad
    arbol["marcas"] = sorted(resultados, key=lambda x: (x.get("marca") or "").lower())

    return arbol

# ============================
# MAIN
# ============================
def main():
    print(f"➡️  Fase 1 | Empresa={EMPRESA} Periodo={PERIODO} | BASE={BASE}")
    esqueleto = construir_esqueleto()

    with open(OUTFILE_JSON, "w", encoding="utf-8") as f:
        json.dump(esqueleto, f, ensure_ascii=False, indent=2)

    print(f"\n✅ JSON jerárquico generado → {OUTFILE_JSON}")
    print(f"   Marcas: {len(esqueleto.get('marcas', []))}")

if __name__ == "__main__":
    main()
