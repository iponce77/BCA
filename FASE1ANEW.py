# Fase1A_playwright.py
# Playwright + login sync previo + calendario multi-página + TODOS los lotes por subasta
# Paginación robusta (URL hasta estancar + fallback a clicks), scroll para lazy-load,
# normalización de URLs de lotes y deduplicado.

import asyncio
import argparse
import os
import re
import urllib.parse
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, BrowserContext, Page

# ========= Zona horaria =========
TZ = None  # Follow runner timezone for Selenium compatibility

# ========= Logging =========
LOG_DIR = Path("./_logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("subastas")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    # Consola
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(sh)
    # Fichero rotativo
    fh = RotatingFileHandler(LOG_DIR / "scraper.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.info("== LOGGER INICIALIZADO ==")
    return logger
LOGGER = setup_logger()
# ========= Configuración de filtros =========
# Excluir por NOMBRE de subasta (regex, case-insensitive). Deja vacío si no quieres excluir por nombre.
BLACKLIST_NAME_PATTERNS = [
    # r"EuroShop",
    # r"Buy\s*Now",
]

# Limitar por país (códigos bandera: ES, FR, BE, IT...). Vacío = todos permitidos.
WHITELIST_COUNTRIES: List[str] = [
    # "ES", "FR"
]

# Excluir por iconos en la tarjeta de subasta (tal y como los pasaste)
FORBIDDEN_ICON_CLASSES = {
    "icon--eauction",
    "icon--liveonline",
    "icon--sealed",
}

# ========= Utilidades =========
def today_ddmm() -> str:
    now = datetime.now()  # naive, runner TZ
    return f"{now.day:02d}/{now.month:02d}"

def should_block(url: str, rtype: str) -> bool:
    """Bloquea recursos pesados para acelerar."""
    url = url.lower()
    if rtype in {"image", "font", "stylesheet", "media"}:
        return True
    return any(url.endswith(ext) for ext in (
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
        ".woff", ".woff2", ".ttf", ".otf", ".css"
    ))

SALE_ID_RE = re.compile(r"saleid_exact:([a-f0-9\-]{36})", re.I)

def parse_icons_from_listing(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    icons = []
    for i in soup.select("i.icon"):
        for cls in (i.get("class") or []):
            if cls.startswith("icon--"):
                icons.append(cls)
    return icons

def parse_listing_block(html: str) -> Dict[str, str]:
    """Lee una tarjeta .listing del calendario y devuelve metadatos de la subasta."""
    soup = BeautifulSoup(html, "html.parser")

    # Enlace + sale_id + nombre
    a = soup.select_one("a.sale-title-link")
    url_subasta = ""
    sale_id = ""
    sale_name = ""
    if a and a.get("href"):
        href = a["href"]
        url_subasta = urllib.parse.urljoin("https://es.bca-europe.com", href)
        m = SALE_ID_RE.search(href)
        if m:
            sale_id = m.group(1)
        sale_name = a.get_text(strip=True)

    # País (flag--XX)
    sale_country = ""
    flag = soup.select_one("i.flag")
    if flag:
        for cls in flag.get("class", []):
            if cls.startswith("flag--") and cls != "flag--left":
                sale_country = cls.split("--", 1)[-1].upper()

    # Fecha visible (Final:/Ends: dd/mm hh:mm)
    sale_date_desc = ""
    date_span = soup.select_one(".listing__date-info span")
    if date_span:
        sale_date_desc = date_span.get_text(" ", strip=True)

    icons = parse_icons_from_listing(html)

    return {
        "sale_id": sale_id,
        "sale_name": sale_name,
        "sale_country": sale_country,
        "sale_date_desc": sale_date_desc,
        "url_subasta": url_subasta,
        "icons": ",".join(icons),
    }

def is_final_today(sale_date_desc: str) -> bool:
    """
    True si contiene 'Final/Fin/Ends/Einde' y el dd/mm de HOY (agnóstico a idioma).
    """
    if not sale_date_desc:
        return False
    text = sale_date_desc.lower()
    if not any(key in text for key in ("final", "fin", "ends", "einde")):
        return False
    return today_ddmm() in text

def name_is_blacklisted(name: str) -> bool:
    if not name:
        return False
    for pat in BLACKLIST_NAME_PATTERNS:
        if re.search(pat, name, flags=re.I):
            return True
    return False

def country_is_allowed(code: str) -> bool:
    if not WHITELIST_COUNTRIES:
        return True
    return code.upper() in WHITELIST_COUNTRIES

def has_forbidden_icons(icon_list_csv: str) -> bool:
    if not icon_list_csv:
        return False
    for cls in icon_list_csv.split(","):
        if cls.strip() in FORBIDDEN_ICON_CLASSES:
            return True
    return False

def extract_page_numbers(html: str) -> List[int]:
    """
    Detecta nº de páginas tanto si hay ?page=N en href como si solo hay
    <a class="nav__link nav__link--pagination">1</a> sin href.
    Devuelve [1..max] o [1] si no hay paginación.
    """
    soup = BeautifulSoup(html, "html.parser")
    pages = set()

    # Caso A: anchors con ?page=N
    for a in soup.select("a.nav__link--pagination[href]"):
        href = a.get("href", "")
        m = re.search(r"[?&]page=(\d+)", href)
        if m:
            pages.add(int(m.group(1)))

    # Caso B: anchors sin href -> usa el texto
    if not pages:
        nums = []
        for a in soup.select("a.nav__link--pagination"):
            t = (a.get_text(strip=True) or "")
            if t.isdigit():
                nums.append(int(t))
        if nums:
            pages = set(range(1, max(nums) + 1))

    return sorted(pages) if pages else [1]

def set_query_param(url: str, key: str, value: str) -> str:
    pr = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qs(pr.query)
    q[key] = [value]
    new_q = urllib.parse.urlencode({k: v[0] for k, v in q.items()})
    return urllib.parse.urlunparse(pr._replace(query=new_q))

# ========= Helpers de paginación/scroll =========
async def auto_scroll(page: Page, steps: int = 3, delta: int = 2500, pause_ms: int = 350):
    """Baja en varios pasos para disparar cargas perezosas."""
    for _ in range(steps):
        await page.mouse.wheel(0, delta)
        await page.wait_for_timeout(pause_ms)

# ========= Fetchers de calendario =========
async def fetch_calendar_page(context: BrowserContext, base_url: str, page_num: int) -> str:
    """Carga el calendario ?page=N y devuelve HTML."""
    url = base_url if page_num == 1 else set_query_param(base_url, "page", str(page_num))
    page = await context.new_page()
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_selector(".listing", timeout=15000)
    html = await page.content()
    await page.close()
    return html

def harvest_calendar_html(html: str) -> List[Dict[str, str]]:
    """Devuelve subastas (dicts) que finalizan hoy, aplicando filtros de nombre, país e iconos."""
    out: List[Dict[str, str]] = []
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select(".listing"):
        block = str(node)
        info = parse_listing_block(block)
        if not info.get("sale_id") or not info.get("url_subasta"):
            continue
        if not is_final_today(info.get("sale_date_desc", "")):
            continue
        if name_is_blacklisted(info["sale_name"]):
            continue
        if not country_is_allowed(info["sale_country"]):
            continue
        if has_forbidden_icons(info.get("icons", "")):
            continue
        out.append(info)
    return out

# ========= Lotes por subasta =========
LOT_HREF_RE = re.compile(r"/Lot\?id=[^\"'\s]+", re.I)

from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

def canonicalize_lot_url(u: str) -> str:
    """
    Normaliza un enlace de lote: conserva solo /Lot y las claves 'id' (+ ItemId/SourceSystem si aparecen).
    Elimina q,bq,sort,page,returnTo,R,SR,promoAppliedSets,saleHeader,cultureCode, etc.
    """
    pr = urlparse(u)
    q = parse_qs(pr.query)

    keep = {}
    if "id" in q:
        keep["id"] = q["id"][0]
    if "ItemId" in q:
        keep["ItemId"] = q["ItemId"][0]       # opcional
    if "SourceSystem" in q:
        keep["SourceSystem"] = q["SourceSystem"][0]  # opcional

    new_q = urlencode(keep)
    return urlunparse((pr.scheme, pr.netloc, "/Lot", pr.params, new_q, ""))

def parse_lot_links_from_listing(html: str, base_host: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls = set()
    for a in soup.select("a[href*='/Lot?id=']"):
        href = a.get("href")
        if not href:
            continue
        full = urllib.parse.urljoin(base_host, href)
        urls.add(canonicalize_lot_url(full))
    # fallback regex si hay tags raros
    if not urls:
        for m in LOT_HREF_RE.finditer(html):
            full = urllib.parse.urljoin(base_host, m.group(0))
            urls.add(canonicalize_lot_url(full))
    return sorted(urls)

async def fetch_sale_lot_page(context: BrowserContext, sale_url: str, page_num: int) -> Tuple[str, str]:
    """Carga el listado de lotes de una subasta en ?page=N y devuelve (html, resolved_url)."""
    url = sale_url if page_num == 1 else set_query_param(sale_url, "page", str(page_num))
    page: Page = await context.new_page()
    await page.goto(url, wait_until="domcontentloaded")
    try:
        await page.wait_for_selector("a[href*='/Lot?id=']", timeout=4000)
    except Exception:
        pass
    await auto_scroll(page, steps=3, delta=3000, pause_ms=350)
    html = await page.content()
    resolved = page.url
    await page.close()
    return html, resolved

async def collect_lots_for_sale(context: BrowserContext, sale_url: str, max_pages: int = 50) -> List[str]:
    """
    Recoge TODOS los lotes de una subasta paginando.
    1) Intenta por URL (?page=N) en bucle hasta que no aparezcan lotes nuevos (estancado).
    2) Si la pág.2 por URL no cambia vs pág.1, fallback a CLICKS (anchors sin href).
    Siempre normaliza y deduplica.
    """
    # ---- Intento por URL “hasta estancar” ----
    seen = set()
    all_urls = set()
    last_hash = None
    url_mode_worked = False

    for pnum in range(1, max_pages + 1):
        html, resolved = await fetch_sale_lot_page(context, sale_url, pnum)
        page_hash = hashlib.md5(html.encode("utf-8", errors="ignore")).hexdigest()

        key = (page_hash, pnum)
        if key in seen:
            break
        seen.add(key)

        pr = urllib.parse.urlparse(resolved)
        base_host = f"{pr.scheme}://{pr.netloc}"
        urls = parse_lot_links_from_listing(html, base_host)
        before = len(all_urls)
        all_urls.update(urls)
        after = len(all_urls)

        if after == before:
            if pnum == 1:
                break
            break
        else:
            url_mode_worked = True

        if pnum == 2 and last_hash == page_hash:
            url_mode_worked = False
            break

        last_hash = page_hash

    if url_mode_worked:
        return sorted(all_urls)

    # ---- Fallback a CLICKS ----
    page: Page = await context.new_page()
    await page.goto(sale_url, wait_until="domcontentloaded")
    try:
        await page.wait_for_selector("a[href*='/Lot?id=']", timeout=4000)
    except Exception:
        pass

    await auto_scroll(page, steps=3, delta=3000, pause_ms=350)
    html1 = await page.content()
    pr = urllib.parse.urlparse(page.url)
    base_host = f"{pr.scheme}://{pr.netloc}"
    all_urls.update(parse_lot_links_from_listing(html1, base_host))

    soup = BeautifulSoup(html1, "html.parser")
    nums = [int((a.get_text(strip=True) or "0"))
            for a in soup.select("a.nav__link--pagination")
            if (a.get_text(strip=True) or "0").isdigit()]
    last = max(nums) if nums else 1
    last = min(last, max_pages)

    prev_count = len(all_urls)
    for pnum in range(2, last + 1):
        btn = page.locator("a.nav__link--pagination", has_text=str(pnum)).first
        if await btn.count() == 0:
            break
        await btn.click()
        await page.wait_for_timeout(300)
        await auto_scroll(page, steps=2, delta=2500, pause_ms=300)
        htmlp = await page.content()
        all_urls.update(parse_lot_links_from_listing(htmlp, base_host))

        if len(all_urls) == prev_count:
            continue
        prev_count = len(all_urls)

    await page.close()
    return sorted(all_urls)

# ========= Exportar Excel por subasta (nuevo Paso 2) =========
async def export_excel_for_sale(context: BrowserContext, sale_id: str, sale_url: str, culture: str, out_dir: Path) -> bool:
    """
    Intenta descargar el Excel de una subasta:
      1) click en a#exportAllExcel con expect_download
      2) fallback: navegar al href del botón con expect_download
      3) fallback final: ReportViewer.axd?bq=saleid_exact:<id>
    Guarda en out_dir/<sale_id>.xls y devuelve True/False.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sale_id}.xls"
    page = await context.new_page()
    try:
        # Asegurar cultureCode en la URL de subasta (si existiera)
        if sale_url and "cultureCode=" not in sale_url:
            sale_url = set_query_param(sale_url, "cultureCode", culture)

        # 1) Ir a la subasta y click → download
        if sale_url:
            await page.goto(sale_url, wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(250)
            btn = page.locator("a#exportAllExcel")
            if await btn.count() > 0:
                try:
                    async with page.expect_download() as di:
                        await btn.click()
                    dl = await di.value
                    await dl.save_as(str(out_file))
                    return True
                except Exception:
                    pass

            # 2) href del botón
            href = await btn.get_attribute("href") if await btn.count() > 0 else None
            if href:
                try:
                    url = urllib.parse.urljoin("https://es.bca-europe.com", href)
                    async with page.expect_download() as di:
                        await page.goto(url)
                    dl = await di.value
                    await dl.save_as(str(out_file))
                    return True
                except Exception:
                    pass

        # 3) ReportViewer directo
        base = "https://es.bca-europe.com/Classic/ReportViewer.axd"
        params = {
            "ShowPricing": "False",
            "sort": "LotNumber",
            "bq": f"saleid_exact:{sale_id}",
            "cultureCode": culture
        }
        rv = f"{base}?{urllib.parse.urlencode(params)}"
        try:
            async with page.expect_download() as di:
                await page.goto(rv)
            dl = await di.value
            await dl.save_as(str(out_file))
            return True
        except Exception:
            return False
    finally:
        if not page.is_closed():
            await page.close()


# ========= Flujo principal =========
async def run(calendar_url: str,
              storage_state: str,
              culture: str,
              cal_max_pages: int,
              stop_on_empty: bool,
              lot_max_pages: int,
              concurrency: int):
    subastas: List[Dict[str, str]] = []
    lotes_rows: List[Dict[str, str]] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        context = await browser.new_context(
            storage_state=storage_state,
            locale=f"{culture}-{culture.upper()}",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        )

        # Bloqueo de recursos
        async def route_handler(route, request):
            if should_block(request.url, request.resource_type):
                await route.abort()
            else:
                await route.continue_()
        await context.route("**/*", route_handler)

        # 1) Recorrer calendario
        for p in range(1, cal_max_pages + 1):
            html = await fetch_calendar_page(context, calendar_url, p)
            found = harvest_calendar_html(html)
            if found:
                subastas.extend(found)
            elif stop_on_empty:
                break

        # Deduplicar por sale_id + normalizar cultureCode
        seen_ids = set()
        subastas_dedup: List[Dict[str, str]] = []
        for s in subastas:
            sid = s.get("sale_id", "")
            if not sid or sid in seen_ids:
                continue
            seen_ids.add(sid)
            if s.get("url_subasta"):
                s["url_subasta"] = set_query_param(s["url_subasta"], "cultureCode", culture)
            subastas_dedup.append(s)
        subastas = subastas_dedup
        LOGGER.info(f"Subastas detectadas hoy (tras filtros): {len(subastas)}")

        # 2) NUEVO: Exportar Todo primero (concurrencia por subasta)
        out_dir = Path("./_output_excels")
        export_failures: List[Dict[str, str]] = []

        sem = asyncio.Semaphore(concurrency)

        async def export_worker(sale: Dict[str, str]):
            async with sem:
                sid = sale.get("sale_id", "")
                sname = sale.get("sale_name", "")
                surl = sale.get("url_subasta", "")
                LOGGER.info(f"Intentando EXPORTAR: [{sname}] ({sid})")
                ok = await export_excel_for_sale(context, sid, surl, culture=culture, out_dir=out_dir)
                if not ok:
                    export_failures.append(sale)
                    LOGGER.warning(f"Export FALLÓ → fallback scraping: [{sname}] ({sid})")
                else:
                    LOGGER.info(f"Export OK: [{sname}] ({sid}) -> _output_excels/{sid}.xls")

        await asyncio.gather(*(export_worker(s) for s in subastas))
        LOGGER.info(f"Export fallidas → fallback scraping: {len(export_failures)}")

        # 2.5) Fallback: SOLO las subastas cuya exportación falló → recolectar lotes (tu Paso 2 original)
        if export_failures:
            sem2 = asyncio.Semaphore(concurrency)

            async def lot_worker(sale: Dict[str, str]):
                async with sem2:
                    try:
                        sale_id = sale.get("sale_id", "")
                        sale_name = sale.get("sale_name", "")
                        sale_url = sale.get("url_subasta", "")
                        LOGGER.info(f"SCRAPING (fallback) lotes: [{sale_name}] ({sale_id})")
                        lot_urls = await collect_lots_for_sale(context, sale_url, max_pages=lot_max_pages)
                        for u in lot_urls:
                            lotes_rows.append({
                                "sale_id": sale_id,
                                "sale_name": sale_name,
                                "lot_url": u
                            })
                        LOGGER.exception(f"Error scraping (fallback) en [{sale.get('sale_name','')}] ({sale.get('sale_id','')})")
                    except Exception as e:
                        lotes_rows.append({
                            "sale_id": sale.get("sale_id", ""),
                            "sale_name": sale.get("sale_name", ""),
                            "lot_url": f"ERROR:{e}"
                        })

            await asyncio.gather(*(lot_worker(s) for s in export_failures))

        await context.close()
        await browser.close()

        # 3) Guardar salidas (dentro de run)
        if subastas:
            print(f"✅ Detectadas {len(subastas)} subastas de hoy (tras filtros/iconos).")
        else:
            print("⚠️ No se detectaron subastas (Final: hoy) tras filtros/iconos).")
        if lotes_rows:
            df_fichas = pd.DataFrame({"link_ficha": [row.get("lot_url") for row in lotes_rows if row.get("lot_url")]})
            if not df_fichas.empty:
                out_name = f"fichas_vehiculos_{datetime.now():%Y%m%d}.xlsx"
                df_fichas.to_excel(out_name, index=False)
                print(f"✅ Guardadas {len(df_fichas)} fichas en {out_name}")
            else:
                print("⚠️ No se captaron URLs de lotes válidas para las subastas encontradas.")
        else:
            print("⛔ No se captaron lotes para las subastas encontradas. No se genera Excel.")
def main():
    from login_poc import perform_login  # sync

    ap = argparse.ArgumentParser(description="Fase 1A (Playwright): calendario multi-página + todos los lotes por subasta")
    ap.add_argument("--calendar-url", required=False, default="https://es.bca-europe.com/buyer/facetedSearch/saleCalendar?bq=&cultureCode=en", help="Calendario BCA (default como Selenium)")
    ap.add_argument("--storage", default="bca_storage_phase1.json")
    ap.add_argument("--culture", default="en")
    ap.add_argument("--calendar-max-pages", type=int, default=5, help="Máx páginas de calendario a revisar")
    ap.add_argument("--stop-on-empty", action="store_true", help="Parar al encontrar una página sin subastas de hoy")
    ap.add_argument("--lot-max-pages", type=int, default=50, help="Límite de páginas de lotes por subasta")
    ap.add_argument("--concurrency", type=int, default=4, help="Subastas en paralelo al recolectar lotes")
    args = ap.parse_args()

    user = os.getenv("BCA_USER")
    pwd  = os.getenv("BCA_PASS")
    if not user or not pwd:
        raise SystemExit("⚠️ Falta definir BCA_USER o BCA_PASS en variables de entorno.")
    perform_login(user=user, password=pwd, headless=True, debug=False,
                  screenshot_on_fail=True, output=Path(args.storage))

    asyncio.run(run(
        calendar_url=args.calendar_url,
        storage_state=args.storage,
        culture=args.culture,
        cal_max_pages=args.calendar_max_pages,
        stop_on_empty=args.stop_on_empty,
        lot_max_pages=args.lot_max_pages,
        concurrency=args.concurrency,
    ))
if __name__ == "__main__":
    main()
