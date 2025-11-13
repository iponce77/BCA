#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase1B_enrich_extra.py
----------------------
Fallback + EnrichExtra (4 campos) con la *misma* tubería de scraping,
clave estable idéntica y escritura in-place sobre el Excel.

Uso (ejemplo):
    python Fase1B_enrich_extra.py -i fichas_vehiculos_20251112.xlsx -c 10 --enrich-extra         --enrich-concurrency 6 --enrich-delay-ms 300 --enrich-jitter-ms 300

Requiere Playwright instalado y con navegadores:
    pip install playwright bs4 pandas openpyxl python-dotenv
    playwright install chromium

Login:
  - Por defecto intenta importar perform_login de login_poc.py (mismo flujo que Fase1A/1B).
  - Si no existe, el stub informará del paso a seguir.
"""
import asyncio, random, re, json, os, sys, unicodedata, urllib.parse, pathlib, time
import logging
import argparse
from typing import Dict, Any, Iterable, Tuple, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext, TimeoutError as PWTimeout

# ====== Intentar reusar tu login existente (login_poc.perform_login) ======
def _fallback_login_stub(user: str, password: str, headless: bool, debug: bool, screenshot_on_fail: bool, output: pathlib.Path):
    raise SystemExit(
        "⚠️ No se encontró 'login_poc.perform_login'. "
        "Sitúa este script junto a login_poc.py o adapta perform_login() aquí."
    )

try:
    # mismo nombre/firmas que ya usas en Fase1A/Fase1B
    from login_poc import perform_login  # type: ignore
except Exception:
    perform_login = _fallback_login_stub  # type: ignore

# ==================== Configuración / Constantes ====================
SEM_LIMIT = 10
MAX_RETRIES = 2
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/125.0.0.0 Safari/537.36")

PREFERRED_DEST = {
    "bastidor": "Bastidor",
    "caja cambios": "Caja de Cambios",
    "pais subasta": "País Subasta",
    "ubicacion": "Ubicación",
}

ENRICH_KEYS_CANON = ["bastidor","caja cambios","pais subasta","ubicacion"]
ENRICH_DEST_NAMES = [PREFERRED_DEST[k] for k in ENRICH_KEYS_CANON]

# Multi-idioma (expandido) → puedes añadir sinónimos si detectas nuevos
_CANON = {
    "bastidor": {
        "bastidor","vin","vehicle identification number","num vin","n vin",
        "chasis","chassis","chassis number","numero de chasis","n chasis",
        "rahmennummer","fahrgestellnummer","numéro de châssis","numero di telaio",
        "frame","frame number"
    },
    "caja cambios": {
        "caja de cambios","transmision","transmisión","transmission","gearbox","gear box",
        "getriebe","schaltgetriebe","automatikgetriebe","boite de vitesses","boîte de vitesses",
        "cambio","tipo de cambio","tipo caja","gear","trans"
    },
    "pais subasta": {
        "pais subasta","país subasta","sale_country","sale country","country","land","pays","paese"
    },
    "ubicacion": {
        "ubicacion","ubicación","location","locatie","standort","emplacement","plaats","ort","lugar","localización"
    },
}

# ==================== Normalización / Canonización ====================
def _norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canon_key(name: str) -> str:
    """Canoniza una etiqueta multi-idioma y devuelve la clave canónica si existe en _CANON."""
    n = _norm(name)
    for k, vs in _CANON.items():
        if n in vs:
            return k
    return n

def deep_iter(obj: Any) -> Iterable[Tuple[str, Any]]:
    """Itera recursivamente (clave, valor) sobre dicts/listas.
    Si encuentra pares tipo {'name':..,'value':..} o {'label':..,'value':..}, emite (name/label, value)."""
    if isinstance(obj, dict):
        lbl = obj.get("label") or obj.get("name") or obj.get("key")
        if lbl is not None and "value" in obj:
            yield (str(lbl), obj["value"])
        for k, v in obj.items():
            yield (str(k), v)
            yield from deep_iter(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from deep_iter(it)
    # valores atómicos → nada

def pick_enrich_fields(res_dict: Dict[str, Any]) -> Dict[str, str]:
    """Extrae únicamente los 4 campos a enriquecer a partir del dict *completo* de scraping,
    usando canonización + PREFERRED_DEST. Aplica fallbacks comunes."""
    out: Dict[str, str] = {}
    for k, v in deep_iter(res_dict):
        if v in (None, "", "nan", "None"):
            continue
        ck = canon_key(k)
        if ck in ENRICH_KEYS_CANON:
            dest = PREFERRED_DEST.get(ck, k)
            out[dest] = str(v).strip()

    # Fallbacks planos típicos que a veces llegan fuera de tablas
    out.setdefault(PREFERRED_DEST["bastidor"],
                   res_dict.get("VIN") or res_dict.get("vin") or res_dict.get("Bastidor") or res_dict.get("bastidor"))
    out.setdefault(PREFERRED_DEST["caja cambios"],
                   res_dict.get("transmission") or res_dict.get("Transmisión") or res_dict.get("Caja de Cambios"))
    out.setdefault(PREFERRED_DEST["pais subasta"],
                   res_dict.get("sale_country") or res_dict.get("País Subasta") or res_dict.get("country"))
    out.setdefault(PREFERRED_DEST["ubicacion"],
                   res_dict.get("location") or res_dict.get("Ubicación"))

    # Limpieza final
    return {k: v for k, v in out.items() if v not in (None, "", "nan", "None")}

# ==================== Clave estable por URL / Resultado ====================
def url_key(u: str) -> str:
    """Clave estable para emparejar scraping <-> filas DF.
      - Preferencia: ItemId (GUID) → 'item:<GUID>'
      - Si no hay, usa id → 'lot:<id>'
      - Último recurso: URL normalizada sin espacios
    """
    if not u:
        return ""
    u = str(u).strip()
    try:
        qs = urllib.parse.urlparse(u).query
        q = urllib.parse.parse_qs(qs)
        item = q.get("ItemId", [None])[0]
        lot  = q.get("id", [None])[0]
        if item:
            return f"item:{item}"
        if lot:
            return f"lot:{lot}"
    except Exception:
        pass
    return re.sub(r"\s+", "", u)

def result_stable_key(res: Dict[str, Any]) -> Optional[str]:
    """Deriva clave estable desde el *resultado* (p.ej. si la URL no traía parámetros)."""
    item = res.get("vehicle_id") or res.get("ItemId") or res.get("itemid")
    lot  = res.get("lot_id") or res.get("id") or res.get("lot")
    if item:
        return f"item:{item}"
    if lot:
        return f"lot:{lot}"
    return None

# ---- Normalización de URL (añade https:// si falta, y host por defecto) ----
def normalize_lot_url(u: str) -> str:
    if not u:
        return u
    u = str(u).strip()
    if u.startswith("http://") or u.startswith("https://"):
        return u
    # //host/....
    if u.startswith("//"):
        return "https:" + u
    # host sin esquema
    if re.match(r"^[\w.-]+/.*", u):
        return "https://" + u
    # ruta absoluta sin host
    if u.startswith("/"):
        return "https://es.bca-europe.com" + u
    # cualquier otra cosa: si parece Lot?id=..., fuerza host
    if u.lower().startswith("lot?") or u.lower().startswith("lot&id"):
        return "https://es.bca-europe.com/" + u
    return u

# ==================== Parsers (idénticos a fallback) ====================
def parse_url_params(url: str) -> Dict[str, Any]:
    qs = urllib.parse.urlparse(url).query
    q = urllib.parse.parse_qs(qs)
    return {
        "lot_id": q.get("id", [None])[0],
        "vehicle_id": q.get("ItemId", [None])[0],
    }

def parse_table(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    out: Dict[str, Any] = {}
    for tbl in soup.select("table.viewlot__table"):
        for tr in tbl.select("tr"):
            tds = [td.get_text(strip=True) for td in tr.select("td")]
            if len(tds) == 2:
                out[tds[0]] = tds[1]
    return out

_DIGITALDATA_PUSH_RE = re.compile(r"digitalData\.push\(\s*(\{.*?\})\s*\);", re.DOTALL | re.IGNORECASE)
_DIGITALDATA_INIT_RE = re.compile(r"window\.digitalData\s*=\s*(\[[^\]]*\]|\{.*?\});", re.DOTALL | re.IGNORECASE)

def parse_digital_data(html: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for blob in _DIGITALDATA_PUSH_RE.findall(html):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        veh = (obj.get("product") or {}).get("vehicle") if isinstance(obj, dict) else None
        if isinstance(veh, dict):
            data.update({
                "make": veh.get("make"),
                "model": veh.get("model"),
                "fuel_type": veh.get("fuelType"),
                "sale_country": veh.get("saleCountry"),
                "sale_code": veh.get("saleCode"),
                "lot": veh.get("lot"),
            })
    for blob in _DIGITALDATA_INIT_RE.findall(html):
        try:
            arr_or_obj = json.loads(blob)
        except Exception:
            continue
        def _walk(o):
            if isinstance(o, dict):
                veh = (o.get("product") or {}).get("vehicle")
                if isinstance(veh, dict):
                    data.update({
                        "make": veh.get("make"),
                        "model": veh.get("model"),
                        "fuel_type": veh.get("fuelType"),
                        "sale_country": veh.get("saleCountry"),
                        "sale_code": veh.get("saleCode"),
                        "lot": veh.get("lot"),
                    })
                for v in o.values():
                    _walk(v)
            elif isinstance(o, list):
                for it in o:
                    _walk(it)
        _walk(arr_or_obj)
    return data

def parse_subheadline(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    span = soup.select_one("span.viewlot__subheadline")
    if not span:
        return {}
    t = span.get_text(" ", strip=True)
    out: Dict[str, Any] = {}
    m = re.search(r"\b(Manual|Autom(?:á|a)tico|DSG|CVT)\b", t, re.IGNORECASE)
    if m: out["transmission"] = m.group(1).capitalize()
    m = re.search(r"([\d\.]+)\s*km\b", t, re.IGNORECASE)
    if m: out["mileage"] = m.group(1).replace(".", "")
    m = re.search(r"\b(Diesel|Gasolina|El[eé]ctrico|H[ií]brido)\b", t, re.IGNORECASE)
    if m: out["fuel_type"] = m.group(1).capitalize()
    return out

def parse_sale_info(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    out: Dict[str, Any] = {}
    panel = soup.select_one("#saleInformationSidePanel .viewlot__saleinfo")
    if panel:
        name = panel.select_one("h6.sale__name__subtitle")
        if name:
            out["sale_name"] = name.get_text(strip=True)
        loc = panel.get_text(" ", strip=True)
        if loc and "location" in loc.lower():
            out.setdefault("location", loc)
    return out

def parse_co2(html: str) -> Dict[str, Any]:
    m = re.search(r"(\d+)\s*(?:g/km|g\s*/\s*km)", html, flags=re.IGNORECASE)
    return {"co2": m.group(1)} if m else {}

# ==================== Fetching (compartido) ====================
async def fetch_lot_html(url: str, context: BrowserContext) -> str:
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=60_000)
        # Mantén una espera corta opcional por la tabla
        try:
            await page.wait_for_selector("table.viewlot__table", timeout=4000)
        except Exception:
            pass
        html = await page.content()
        try:
            tbl = await page.evaluate("() => document.querySelector('table.viewlot__table')?.outerHTML")
            if tbl:
                html += "\n" + tbl
        except Exception:
            pass
        return html
    finally:
        await page.close()

async def bound_fetch(sem: asyncio.Semaphore, context: BrowserContext, url: str) -> Dict[str, Any]:
    async with sem:
        try:
            html = await asyncio.wait_for(fetch_lot_html(url, context), timeout=75)
            data: Dict[str, Any] = {}
            data.update(parse_digital_data(html))
            data.update(parse_subheadline(html))
            data.update(parse_sale_info(html))
            data.update(parse_co2(html))
            data.update(parse_table(html))
            res = parse_url_params(url)
            res["viewlot_url"] = url
            res.update(data)
            return res
        except Exception as e:
            return {"viewlot_url": url, "error": str(e)}

# ==================== Utilidades DF ====================
def _is_empty_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        t = s.astype(str).str.strip().str.lower()
        return s.isna() | t.eq("") | t.eq("nan") | t.eq("none")
    else:
        return s.isna()

def ensure_dest_columns(df: pd.DataFrame) -> None:
    for dest in ENRICH_DEST_NAMES:
        if dest not in df.columns:
            df[dest] = pd.Series([None]*len(df), dtype="object")
        else:
            df[dest] = df[dest].astype("object")

def detect_link_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in {"ver vínculo del lote","ver vinculo del lote","link_ficha","link","url","enlace"}:
            return c
    raise ValueError("No encuentro la columna de enlace (p.ej. 'Ver vínculo del lote').")

def resolve_target_column(scraped_col: str, df_columns: List[str]) -> str:
    target_key = canon_key(scraped_col)
    for c in df_columns:
        if canon_key(c) == target_key:
            return c
    return PREFERRED_DEST.get(target_key, scraped_col)

# ==================== Workers: Fallback & EnrichExtra ====================
async def run_fallback(context: BrowserContext, df: pd.DataFrame, link_col: str, fallback_mask: pd.Series, concurrency: int, log: logging.Logger) -> Dict[str, Dict[str, Any]]:
    urls_fallback = df.loc[fallback_mask, link_col].astype(str).tolist()
    log.info(f"Comenzando scraping SOLO fallback ({len(urls_fallback)} lotes) con concurrencia={concurrency}")

    resultados_by_key: Dict[str, Dict[str, Any]] = {}
    resultados_dict: Dict[str, Dict[str, Any]] = {}

    sem = asyncio.Semaphore(concurrency)
    tasks = [bound_fetch(sem, context, normalize_lot_url(u)) for u in urls_fallback]
    done = 0
    for coro in asyncio.as_completed(tasks):
        res = await coro; done += 1
        u = res.get("viewlot_url", "–")
        if res.get("error"):
            log.warning(f"[{done}/{len(urls_fallback)}] ERROR {u}: {res['error']}")
        else:
            log.info(f"[{done}/{len(urls_fallback)}] Completado {u}")
        k = url_key(u)
        resultados_dict[u] = res
        resultados_by_key[k] = res
        k2 = result_stable_key(res)
        if k2:
            resultados_by_key[k2] = res

    for retry in range(1, MAX_RETRIES+1):
        fallidas = [u for u, d in resultados_dict.items() if d.get("error")]
        if not fallidas:
            log.info(f"Todos los lotes han sido scrapeados correctamente tras {retry-1} reintentos.")
            break
        log.warning(f"Reintentando {len(fallidas)} lotes (intento {retry}/{MAX_RETRIES})")
        r_tasks = [bound_fetch(sem, context, normalize_lot_url(u)) for u in fallidas]
        for coro in asyncio.as_completed(r_tasks):
            res = await coro
            u = res.get("viewlot_url", "–")
            k = url_key(u)
            resultados_dict[u] = res
            resultados_by_key[k] = res
            k2 = result_stable_key(res)
            if k2:
                resultados_by_key[k2] = res
            if res.get("error"):
                log.warning(f"[Retry {retry}] ERROR {u}: {res['error']}")
            else:
                log.info(f"[Retry {retry}] Recuperado {u}")

    return resultados_by_key

def _collect_canon_labels(res_dict: Dict[str, Any], limit: int = 8) -> List[str]:
    seen = []
    for k, _ in deep_iter(res_dict):
        ck = canon_key(k)
        if ck and ck not in seen:
            seen.append(ck)
        if len(seen) >= limit:
            break
    return seen

async def run_enrich_extra(context: BrowserContext,
                           df: pd.DataFrame,
                           link_col: str,
                           fallback_mask: pd.Series,
                           enrich_concurrency: int,
                           enrich_delay_ms: int,
                           enrich_jitter_ms: int,
                           log: logging.Logger) -> Dict[str, Dict[str, str]]:
    ensure_dest_columns(df)

    nonfallback_mask = ~fallback_mask
    need_any = None
    for dest in ENRICH_DEST_NAMES:
        col_need = _is_empty_series(df[dest])
        need_any = col_need if need_any is None else (need_any | col_need)

    extra_mask = (nonfallback_mask & need_any).fillna(False)
    urls_extra = df.loc[extra_mask, link_col].astype(str).tolist()

    seen_keys = set()
    urls_extra_unique: List[str] = []
    for u in urls_extra:
        k = url_key(u)
        if k not in seen_keys:
            seen_keys.add(k); urls_extra_unique.append(u)

    log.info(f"[EnrichExtra] Filas a enriquecer: {int(extra_mask.sum())} | URLs únicas: {len(urls_extra_unique)}")

    sem_e = asyncio.Semaphore(max(1, int(enrich_concurrency)))
    base_delay = max(0, enrich_delay_ms) / 1000.0
    jitter = max(0, enrich_jitter_ms) / 1000.0

    results_extra_by_key: Dict[str, Dict[str, str]] = {}
    diagnostics: Dict[str, List[str]] = {}

    done = 0
    total = len(urls_extra_unique)
    async def worker(u: str):
        nonlocal done
        await asyncio.sleep(base_delay + random.uniform(0, jitter))
        res = await bound_fetch(sem_e, context, normalize_lot_url(u))
        k = url_key(u)
        if not res or res.get("error"):
            results_extra_by_key[k] = {}
            diagnostics[k] = ["<error> " + str(res.get('error'))]
            done += 1
            if done % 100 == 0 or done == total:
                log.info(f"[EnrichExtra] Progreso: {done}/{total}")
            return
        mini = pick_enrich_fields(res)
        results_extra_by_key[k] = {k2: v2 for k2, v2 in mini.items()
                                   if v2 not in (None, "", "nan", "None")}
        diagnostics[k] = _collect_canon_labels(res, limit=8)
        k2 = result_stable_key(res)
        if k2:
            results_extra_by_key[k2] = results_extra_by_key[k]
            diagnostics[k2] = diagnostics[k]
        done += 1
        if done % 100 == 0 or done == total:
            log.info(f"[EnrichExtra] Progreso: {done}/{total}")

    await asyncio.gather(*(worker(u) for u in urls_extra_unique))

    got = sum(1 for v in results_extra_by_key.values() if v)
    log.info(f"[EnrichExtra] URLs con datos válidos: {got}/{len(urls_extra_unique)}")
    if len(urls_extra_unique) > 0 and got == 0:
        log.warning("[EnrichExtra] ⚠️ 0 URLs con datos. Muestras de etiquetas canonizadas vistas:")
        count = 0
        for k in list(diagnostics.keys())[:3]:
            labels = diagnostics.get(k, [])
            log.warning(f" - key={k} → etiquetas={labels}")
            count += 1
            if count >= 3:
                break

    return results_extra_by_key

# ==================== Escritura en DF ====================
def write_fallback_to_df(df: pd.DataFrame,
                         resultados_by_key: Dict[str, Dict[str, Any]],
                         link_col: str,
                         fallback_mask: pd.Series,
                         log: logging.Logger) -> None:
    campos = set()
    for d in resultados_by_key.values():
        if not isinstance(d, dict):
            continue
        campos.update(d.keys())
    campos -= {"viewlot_url","error","lot_id","vehicle_id"}

    columnas_existentes = list(df.columns)
    destino_por_campo = {}
    for col_scraped in sorted(campos):
        dest = resolve_target_column(col_scraped, columnas_existentes)
        destino_por_campo[col_scraped] = dest
        if dest not in df.columns:
            df[dest] = pd.Series([None]*len(df), dtype="object")
            columnas_existentes.append(dest)
        else:
            df[dest] = df[dest].astype("object")

    key_series_fb = df[link_col].map(url_key)
    for col_scraped, col_dest in destino_por_campo.items():
        mapped_fb = key_series_fb.map(lambda k: resultados_by_key.get(k, {}).get(col_scraped)).astype("object")
        df.loc[fallback_mask, col_dest] = mapped_fb[fallback_mask].values

    if 'error_fase1b' not in df.columns:
        df['error_fase1b'] = pd.Series([None]*len(df), dtype="object")
    else:
        df['error_fase1b'] = df['error_fase1b'].astype("object")
    mapped_err = key_series_fb.map(lambda k: resultados_by_key.get(k, {}).get('error')).astype("object")
    df.loc[fallback_mask, 'error_fase1b'] = mapped_err[fallback_mask].values

def write_enrich_to_df(df: pd.DataFrame,
                       results_extra_by_key: Dict[str, Dict[str, str]],
                       link_col: str,
                       fallback_mask: pd.Series,
                       log: logging.Logger) -> None:
    ensure_dest_columns(df)
    key_series = df[link_col].map(url_key)
    nonfallback_mask = ~fallback_mask

    def _has_value(x: Any) -> bool:
        if x is None:
            return False
        s = str(x).strip().lower()
        return s not in ("", "nan", "none")

    for dest in ENRICH_DEST_NAMES:
        mapped_all = key_series.map(lambda k: results_extra_by_key.get(k, {}).get(dest))
        has_val = mapped_all.map(_has_value).fillna(False)
        target_mask = (nonfallback_mask & has_val).fillna(False)
        if target_mask.any():
            df.loc[target_mask, dest] = mapped_all[target_mask].astype("object").values

    log.info("[EnrichExtra] Escritura completada en DF.")

# ==================== Flujo principal ====================
async def main_async(input_excel: str, storage_state: str, concurrency: int,
                     enrich_extra: bool, enrich_concurrency: int,
                     enrich_delay_ms: int, enrich_jitter_ms: int):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    log = logging.getLogger("fase1b")
    log.info("=== INICIO Fase1B (fallback + EnrichExtra) ===")

    df = pd.read_excel(input_excel)
    df = df.reset_index(drop=True)

    link_col = detect_link_col(df)

    nonlink_cols = [c for c in df.columns if c != link_col]
    empty_mask = None
    for c in nonlink_cols:
        ce = _is_empty_series(df[c])
        empty_mask = ce if empty_mask is None else (empty_mask & ce)
    fallback_mask = empty_mask.fillna(True)

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=True)
        context: BrowserContext = await browser.new_context(
            storage_state=storage_state, locale="es-ES", user_agent=USER_AGENT
        )
        # --- BLOQUEO DE RECURSOS PESADOS / TERCIARIOS ---
        async def _route_block(route, request):
            rt = request.resource_type
            url = request.url
            # Bloquea imágenes, media, fuentes, hojas de estilo y analytics conocidos
            if rt in {"image","media","font","stylesheet"}:
                return await route.abort()
            if any(h in url for h in [
                "google-analytics", "googletagmanager", "doubleclick.net",
                "hotjar", "facebook", "optimizely", "segment", "newrelic"
            ]):
                return await route.abort()
            return await route.continue_()
        await context.route("**/*", _route_block)
        try:
            p = await context.new_page()
            await p.goto("https://es.bca-europe.com", timeout=30_000)
            await p.close()
        except Exception as e:
            log.warning(f"Warm-up Chromium falló: {e}")

        resultados_by_key = await run_fallback(context, df, link_col, fallback_mask, concurrency, log)

        results_extra_by_key: Dict[str, Dict[str, str]] = {}
        if enrich_extra:
            results_extra_by_key = await run_enrich_extra(
                context, df, link_col, fallback_mask,
                enrich_concurrency=enrich_concurrency,
                enrich_delay_ms=enrich_delay_ms,
                enrich_jitter_ms=enrich_jitter_ms,
                log=log
            )

        await context.close()
        await browser.close()

    write_fallback_to_df(df, resultados_by_key, link_col, fallback_mask, log)
    if enrich_extra and results_extra_by_key:
        write_enrich_to_df(df, results_extra_by_key, link_col, fallback_mask, log)

    df.to_excel(input_excel, index=False)
    logging.getLogger("fase1b").info(f"Excel enriquecido (inplace) guardado en {input_excel}")

def main():
    parser = argparse.ArgumentParser(description="Fase 1B: fallback + EnrichExtra (inplace)")
    parser.add_argument("-i","--input", required=True, help="Excel de entrada/salida (p. ej. fichas_vehiculos_YYYYMMDD.xlsx)")
    parser.add_argument("-s","--storage", default="bca_storage_phase1.json", help="Playwright storage_state JSON")
    parser.add_argument("-c","--concurrency", type=int, default=SEM_LIMIT, help="Concurrencia fallback")
    parser.add_argument("--enrich-extra", action="store_true", help="Enriquecer 4 campos (Bastidor, Caja de Cambios, País Subasta, Ubicación)")
    parser.add_argument("--enrich-concurrency", type=int, default=6, help="Concurrencia enrich-extra")
    parser.add_argument("--enrich-delay-ms", type=int, default=300, help="Delay base enrich-extra (ms)")
    parser.add_argument("--enrich-jitter-ms", type=int, default=300, help="Jitter enrich-extra (ms)")
    args = parser.parse_args()

    user = os.getenv("BCA_USER"); password = os.getenv("BCA_PASS")
    if not user or not password:
        sys.exit("⚠️  Define BCA_USER y BCA_PASS en variables de entorno.")

    perform_login(
        user, password,
        headless=True, debug=False, screenshot_on_fail=True,
        output=pathlib.Path(args.storage)
    )

    asyncio.run(main_async(
        input_excel=args.input,
        storage_state=args.storage,
        concurrency=args.concurrency,
        enrich_extra=args.enrich_extra,
        enrich_concurrency=args.enrich_concurrency,
        enrich_delay_ms=args.enrich_delay_ms,
        enrich_jitter_ms=args.enrich_jitter_ms
    ))

if __name__ == "__main__":
    main()
