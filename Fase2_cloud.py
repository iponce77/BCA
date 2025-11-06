#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase2_cloud.py · Fase 2 (Plug & Play · Login garantizado · Alto rendimiento)

• Login SIEMPRE al inicio (como en el script antiguo), imprime el bloque de login.
• Cookies cargadas a requests.Session y reusadas en un ÚNICO contexto Playwright.
• Pool de páginas (reutilización) => mucho menos overhead que crear context/page por ficha.
• Captura de XHR flexible (no exige 2xx) + fallback fetch() in‑page.
• Camino rápido con requests + JSON tolerant (aunque content-type sea text/plain).
• Control de rachas (50 fallos) => relogin + cooldown y se resetea.
• Concurrencia por defecto 8 (sube velocidad, ajustable con --max-pages).
• Merge in-place sobre el Excel de entrada con garantías de columnas.
"""

import os
import sys
import json
import time
import random
import logging
import asyncio
import pathlib
import argparse
import urllib.parse as ul
from datetime import datetime
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from playwright.async_api import async_playwright, TimeoutError as PWTimeout, Error as PWError
from auth_playwright import get_session
from login_poc import perform_login

# ────────────────────────────── Config global ──────────────────────────────
load_dotenv()

LOGIN_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
DEFAULT_STORAGE = pathlib.Path("bca_storage.json")
BID_RE          = "/api/lot/bidpanel"
MAX_RETRIES     = 1  # intentos extra (además del primer pase)

# Session de requests + encabezado coherente
SESSION    = get_session(DEFAULT_STORAGE)
USER_AGENT = SESSION.headers.get("User-Agent", LOGIN_UA)
SESSION.headers.update({"User-Agent": USER_AGENT})

# Mapa SourceSystem conocido
SRC_MAP = {"buyergateway": "1", "peep": "3"}


# ────────────────────────────── Utilidades login ──────────────────────────────
def _print_login_banner(msg: str) -> None:
    print("=" * 72)
    print(msg)
    print("=" * 72)


def force_login(state_path: pathlib.Path):
    """Login síncrono (idéntico al antiguo flujo) y guarda storage_state en JSON.

    Si estamos *dentro* de un asyncio loop, ejecuta `perform_login` en un
    thread separado para evitar usar sync_playwright() en la hebra del loop.
    """
    _print_login_banner("→ Ejecutando LOGIN inicial (forzado)")
    user, pwd = os.getenv("BCA_USER"), os.getenv("BCA_PASS")
    if not user or not pwd:
        raise RuntimeError("BCA_USER / BCA_PASS no definidos (env o .env)")

    def _call_login():
        perform_login(
            user=user,
            password=pwd,
            headless=True,
            debug=False,
            screenshot_on_fail=False,
            output=state_path,
        )

    # ¿Hay un loop corriendo en esta hebra?
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Ejecutar en hilo separado y propagar excepciones
        import threading
        exc = {}

        def _target():
            try:
                _call_login()
            except Exception as e:
                exc["e"] = e

        t = threading.Thread(target=_target, daemon=False)
        t.start()
        t.join()
        if "e" in exc:
            raise exc["e"]
    else:
        # Llamada síncrona "normal" (antes de asyncio.run)
        _call_login()

    print(f"→ Login completado, storage generado: {state_path}")


def _load_cookies_into_session(state_path: pathlib.Path) -> int:
    """Carga cookies del storage_state JSON (Playwright) en SESSION (requests)."""
    print("→ Cargando cookies al requests.Session")
    SESSION.cookies.clear()

    if not state_path.exists():
        print(f"ℹ️ No existe storage_state: {state_path} (Session sin cookies)")
        return 0

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        state = eval(state_path.read_text(encoding="utf-8"))  # retrocompat

    count = 0
    for ck in state.get("cookies", []):
        dom = (ck.get("domain") or "").lstrip(".")
        name = ck.get("name"); value = ck.get("value"); path = ck.get("path") or "/"
        if not (dom and name and value):
            continue
        SESSION.cookies.set(name, value, domain=dom, path=path); count += 1
        # armonizar subdominios típicos
        if dom.endswith("bca.com"):
            SESSION.cookies.set(name, value, domain="es.bca-europe.com", path=path); count += 1
        if dom.endswith("bca-europe.com"):
            SESSION.cookies.set(name, value, domain="es.bca-europe.com", path=path); count += 1

    print(f"→ Cookies cargadas ({count})")
    return count


async def ensure_storage(path: pathlib.Path):
    """Verifica existencia de storage; si no existe, fuerza login."""
    if not path.exists():
        force_login(path)
    # recargar cookies en SESSION siempre (para consistencia)
    _load_cookies_into_session(path)


# ────────────────────────────── Helpers bidpanel ──────────────────────────────
def mapped_src(src_raw: str) -> str:
    s = (src_raw or "").strip()
    return s if s.isdigit() else SRC_MAP.get(s.lower(), "0")


def build_bidpanel_url(link: str, src_override: Optional[str] = None) -> str:
    q       = ul.parse_qs(ul.urlsplit(link).query)
    lot_id  = q.get("id", [""])[0]
    src_raw = q.get("SourceSystem", q.get("sourceSystem", ["0"]))[0]
    src     = src_override or mapped_src(src_raw)
    return f"https://es.bca-europe.com/api/lot/bidpanel?id={lot_id}&sourceSystem={src}"


def get_bidpanel_direct(link: str, timeout=10, debug_net=False) -> Optional[dict]:
    """Intento directo con requests (rápido). Relajado en content-type y prueba varias sourceSystem."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": link,
        "Origin": "https://es.bca-europe.com",
        "Accept-Language": "es-ES,es;q=0.9",
    }
    tried: List[Optional[str]] = []
    for src in [None, "3", "1", "0", "2"]:  # prioriza PEEP y BuyerGateway
        if src in tried:
            continue
        tried.append(src)
        url = build_bidpanel_url(link, src_override=src)
        try:
            r = SESSION.get(url, timeout=timeout, headers=headers)
            if debug_net:
                logging.debug("↩ %s → %s (%s)", url, r.status_code, r.headers.get("content-type",""))
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    text = r.text.strip()
                    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
                        return json.loads(text)
        except Exception as e:
            if debug_net:
                logging.debug("✖ petición %s falló: %s", url, e)
            continue
    return None


async def grab_bidpanel_json(page, link: str, timeout_ms=14000, debug=False):
    """Captura XHR flexible + fallback fetch() in-page con cookies de contexto."""
    url_api = build_bidpanel_url(link)
    try:
        async with page.expect_response(lambda r: BID_RE in r.url.lower(), timeout=timeout_ms) as resp_info:
            await page.goto(link, wait_until="domcontentloaded")
        resp = await resp_info.value
        if debug:
            try:
                logging.debug("[xhr] %s %s", resp.status, resp.url)
            except Exception:
                pass
        if resp.ok:
            try:
                return await resp.json()
            except Exception:
                txt = await resp.text()
                if txt.strip().startswith("{"):
                    return json.loads(txt)
                return None
        # Fallback: fetch() in‑page (misma sesión/cookies)
        js = """
            (u) => fetch(u, {credentials:'include'})
                    .then(r => r.ok ? r.json() : null)
                    .catch(() => null)
        """
        return await page.evaluate(js, url_api)
    except PWTimeout:
        # último intento: fetch() directo si el XHR no llegó
        try:
            js = """
                (u) => fetch(u, {credentials:'include'})
                        .then(r => r.ok ? r.json() : null)
                        .catch(() => null)
            """
            return await page.evaluate(js, url_api)
        except PWError:
            return None
    except PWError:
        return None


FIELDS = [
    "lot_id","auction_name","end_date","starting_bid","current_bid","winning_bid",
    "buy_now_price","currency","lot_status","lot_bidding_status","sale_status",
    "number_of_bids","vat_type","vat_rate","luxury_tax"
]

def extract_fields(raw: dict, link: str) -> dict:
    return {
        "link_ficha":           link,
        "lot_id":               raw.get("lotId") or raw.get("vehicleId"),
        "auction_name":         raw.get("auctionName") or raw.get("saleName"),
        "end_date":             raw.get("endDate"),
        "starting_bid":         raw.get("startingBidAmount"),
        "current_bid":          raw.get("currentBidAmount"),
        "winning_bid":          raw.get("winningBid"),
        "buy_now_price":        raw.get("buyNowAmount"),
        "currency":             raw.get("currencyCode"),
        "lot_status":           raw.get("lotStatus"),
        "lot_bidding_status":   raw.get("lotBiddingStatus"),
        "sale_status":          raw.get("saleStatus"),
        "number_of_bids":       raw.get("numberOfBids"),
        "vat_type":             raw.get("vatTypeDescription"),
        "vat_rate":             raw.get("vatRate"),
        "luxury_tax":           raw.get("HasLuxuryTax"),
    }


# ────────────────────────────── Runner ──────────────────────────────
async def run(args):
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # 0) LOGIN explícito SIEMPRE (como pediste)
    await ensure_storage(args.storage)
    print("→ Login OK. Continuando con scraping…")

    # 1) Carga Excel
    print("→ Leyendo Excel de entrada:", args.excel_in)
    df_in = pd.read_excel(args.excel_in)
    print("→ Excel leído. Shape:", df_in.shape)

    if "link_ficha" not in df_in.columns:
        raise SystemExit("ERROR: no existe columna 'link_ficha' en el Excel de entrada.")

    links = df_in["link_ficha"].dropna().astype(str).tolist()
    print("→ Links a scrapear:", len(links))

    # 2) Playwright (browser + contexto único + pool de páginas)
    print("→ Iniciando Playwright...")
    async with async_playwright() as p:
        print("→ Playwright iniciado")
        browser = await p.chromium.launch(
            headless=not args.visible,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--window-size=1920,1080",
            ],
        )
        print("→ Browser lanzado")

        # contexto inicial
        ctx = await browser.new_context(storage_state=str(args.storage),
                                        locale="es-ES", user_agent=LOGIN_UA)
        # usamos un contenedor para manejar swap de contexto de forma segura
        nonlocal_ctx = {"ctx": ctx}

        # Crear pool de páginas reutilizables
        page_pool: asyncio.Queue = asyncio.Queue()
        for _ in range(args.max_pages):
            page = await nonlocal_ctx["ctx"].new_page()
            page_pool.put_nowait(page)

        sem = asyncio.Semaphore(args.max_pages)
        relogin_lock = asyncio.Lock()   # <<-- LOCK para proteger uso de páginas y relogin
        results = []
        failure_streak = 0

        async def worker(lk: str):
            nonlocal failure_streak
            rec = None

            # Camino rápido con requests (no consume "página" si funciona)
            try:
                data = get_bidpanel_direct(lk, timeout=args.req_timeout_s, debug_net=args.debug_net)
                if data:
                    rec = extract_fields(data, lk)
            except Exception as e:
                logging.debug("direct error %s: %s", lk, e)

            if rec is None:
                # Protección: antes de tomar/usar una página, adquirir relogin_lock
                async with relogin_lock:
                    async with sem:
                        page = await page_pool.get()
                        try:
                            # Jitter para desincronizar rafagas
                            await asyncio.sleep(random.uniform(0.02, 0.12))
                            data = await grab_bidpanel_json(page, lk, timeout_ms=args.xhr_timeout_ms, debug=args.debug_net)
                            if data:
                                rec = extract_fields(data, lk)
                            else:
                                logging.warning("❌ %s → BidPanel no encontrado", lk)
                                rec = {"link_ficha": lk, "error": "BidPanel no encontrado"}
                        except Exception as e:
                            rec = {"link_ficha": lk, "error": f"fallback:{e}"}
                        finally:
                            # Reusar la página (si el contexto se ha cerrado, la página ya estará cerrada;
                            # el pool será reconstruido desde el relogin que controla la misma lock)
                            try:
                                page_pool.put_nowait(page)
                            except Exception:
                                pass

            # Streak breaker
            if rec.get("error"):
                failure_streak += 1
            else:
                failure_streak = 0

            # Si toca relogin: hacerlo dentro del relogin_lock para que nadie use páginas
            if failure_streak >= args.relogin_streak:
                logging.warning("⚠️ %d fallos consecutivos → relogin + cooldown", failure_streak)
                async with relogin_lock:
                    # cerrar contexto antiguo
                    try:
                        await nonlocal_ctx["ctx"].close()
                    except Exception:
                        pass

                    # Forzar login (se ejecutará en hilo si estamos dentro del loop)
                    force_login(args.storage)
                    _load_cookies_into_session(args.storage)

                    # crear nuevo contexto y reconstruir pool
                    ctx2 = await browser.new_context(storage_state=str(args.storage),
                                                     locale="es-ES", user_agent=LOGIN_UA)
                    # cerrar páginas antiguas en la cola si quedan y vaciar
                    try:
                        while not page_pool.empty():
                            try:
                                pg = page_pool.get_nowait()
                                try:
                                    await pg.close()
                                except Exception:
                                    pass
                            except Exception:
                                break
                    except Exception:
                        pass

                    # rellenar nuevo pool
                    for _ in range(args.max_pages):
                        page = await ctx2.new_page()
                        page_pool.put_nowait(page)

                    # swap context
                    nonlocal_ctx["ctx"] = ctx2
                    failure_streak = 0

            results.append(rec)

        print("→ Lanzando scraping con pool de páginas…")
        await asyncio.gather(*(worker(l) for l in links))


        # Reintentos de fallidas (rápidos, mismo contexto)
        for retry in range(1, MAX_RETRIES + 1):
            fallidas = [r["link_ficha"] for r in results if r.get("error")]
            if not fallidas:
                break
            logging.warning("Reintento #%d de %d → %d fallidas", retry, MAX_RETRIES, len(fallidas))
            retry_results = []

            async def r_worker(lk: str):
                page = await page_pool.get()
                try:
                    await asyncio.sleep(random.uniform(0.02, 0.12))
                    data = await grab_bidpanel_json(page, lk, timeout_ms=args.xhr_timeout_ms, debug=args.debug_net)
                    if data:
                        retry_results.append(extract_fields(data, lk))
                    else:
                        retry_results.append({"link_ficha": lk, "error": "BidPanel no encontrado"})
                finally:
                    page_pool.put_nowait(page)

            await asyncio.gather(*(r_worker(l) for l in fallidas))

            # substituir
            patch = {r["link_ficha"]: r for r in retry_results}
            for i, r in enumerate(results):
                k = r["link_ficha"]
                if k in patch:
                    results[i] = patch[k]

        # Cierre ordenado
        try:
            while not page_pool.empty():
                pg = await page_pool.get()
                await pg.close()
        except Exception:
            pass
        await nonlocal_ctx["ctx"].close()
        await browser.close()

    # 3) Merge/export
    print("→ Haciendo merge inplace con Excel de entrada")
    df_tmp = pd.DataFrame(results)
    if "error" not in df_tmp.columns:
        df_tmp["error"] = None
    for col in FIELDS:
        if col not in df_tmp.columns:
            df_tmp[col] = None

    df_fase2 = df_tmp[["link_ficha", *FIELDS, "error"]]
    df_merged = df_in.merge(df_fase2, on="link_ficha", how="left", suffixes=("", "_fase2"))

    for col in FIELDS + ["error"]:
        if col in df_merged.columns and col + "_fase2" in df_merged.columns:
            df_merged[col] = df_merged[col].combine_first(df_merged[col + "_fase2"])
            df_merged.drop(columns=[col + "_fase2"], inplace=True, errors="ignore")
        elif col not in df_merged.columns and col in df_fase2.columns:
            df_merged[col] = df_fase2[col]

    df_merged.to_excel(args.excel_in, index=False)
    print(f"→ Excel pipeline actualizado in‑place → {args.excel_in}")


# ────────────────────────────── CLI ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--excel-in",  type=pathlib.Path, required=True)
    ap.add_argument("-s","--storage",   type=pathlib.Path, default=DEFAULT_STORAGE)
    ap.add_argument("-p","--max-pages", type=int, default=10, help="Páginas/playwright concurrentes (pool)")
    ap.add_argument("--xhr-timeout-ms", type=int, default=11000)
    ap.add_argument("--req-timeout-s",  type=int, default=8)
    ap.add_argument("--relogin-streak", type=int, default=50, help="Racha de fallos para relogin")
    ap.add_argument("--visible", action="store_true")
    env_debug = os.getenv("F2_DEBUG_NET", "false").lower() == "true"
    ap.add_argument("--debug-net", action="store_true", default=env_debug)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    # Forzar login inicial SIEMPRE (como pidió el usuario)
    os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)  # evitar rutas raras en runners
    force_login(args.storage)
    _load_cookies_into_session(args.storage)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

