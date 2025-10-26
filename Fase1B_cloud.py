import asyncio
import logging
import argparse
import urllib.parse
import re
import json
import pandas as pd
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext
from dotenv import load_dotenv
load_dotenv()
import os
import pathlib
import sys

# Importa perform_login del login_poc
from login_poc import perform_login

SEM_LIMIT = 10  # Número de fetches paralelos
MAX_RETRIES = 2  # Número de reintentos para fichas fallidas

MARCA_MAP = {
    "ALFA": "ALFA ROMEO", "ALFA ROMEO": "ALFA ROMEO",
    "BMWI": "BMW", "BMW": "BMW",
    "CAN AM": "CAN-AM", "CAN-AM": "CAN-AM",
    "DFM": "DFSK", "DFSK": "DFSK",
    "KGM": "SSANGYONG", "SSANGYONG": "SSANGYONG",
    "LAND": "LAND ROVER", "LAND ROVER": "LAND ROVER",
    "LYNK": "LYNK&CO", "LYNK & CO": "LYNK&CO", "LYNK&CO": "LYNK&CO",
    "MERCEDES": "MERCEDES-BENZ", "MERCEDES-BENZ": "MERCEDES-BENZ",
    **{k: k for k in [
        "ACCESS", "AIWAYS", "APRILIA", "ASTON MARTIN", "AUDI", "BARTHAU", "BYD", "CATERPILLAR",
        "CHEVROLET", "CHRYSLER", "CITROEN", "CROWN", "CUPRA", "DACIA", "DETHLEFFS", "DODGE", "DS",
        "DUCATI", "FIAT", "FORD", "GENESIS", "GWM", "HONDA", "HYUNDAI", "IVECO", "JAGUAR", "JEEP",
        "JUNGHEINRICH", "KIA", "KNAUS", "LANCIA", "LEXUS", "LIGIER", "LINDE", "LOTUS", "MAN",
        "MASERATI", "MAXUS", "MAZDA", "MG", "MINI", "MITSUBISHI", "NISSAN", "OPEL", "PEUGEOT",
        "PIAGGIO", "POLESTAR", "PORSCHE", "RENAULT", "SAAB", "SEAT", "SKODA", "SMART", "STIHL",
        "SUBARU", "SUZUKI", "TESLA", "TOYOTA", "VESPA", "VOLKSWAGEN", "VOLVO", "XPENG", "YAMAHA"
    ]}
}

def parse_url_params(url: str) -> dict:
    qs = urllib.parse.urlparse(url).query
    params = urllib.parse.parse_qs(qs)
    return {
        "lot_id": params.get("id", [None])[0],
        "sale_id": None,
        "vehicle_id": params.get("ItemId", [None])[0],
    }

def parse_saleid_from_bq(url: str) -> str:
    qs = urllib.parse.urlparse(url).query
    for part in qs.split("&"):
        if part.startswith("bq="):
            v = urllib.parse.unquote_plus(part[3:])
            if v.startswith("saleid_exact:"):
                return v.split(":", 1)[1]
    return None

def parse_table(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    datos = {}
    tabla = soup.select_one("table.viewlot__table")
    if tabla:
        for row in tabla.select("tr"):
            cols = [td.get_text(strip=True) for td in row.select("td")]
            if len(cols) == 2:
                datos[cols[0]] = cols[1]
    return datos

def parse_digital_data(html: str) -> dict:
    matches = re.findall(r"digitalData\.push\(\s*(\{.*?\})\s*\);", html, flags=re.DOTALL)
    for jsobj in matches:
        try:
            obj = json.loads(jsobj)
        except json.JSONDecodeError:
            continue
        veh = obj.get("product", {}).get("vehicle")
        if isinstance(veh, dict):
            return {
                "make": veh.get("make"),
                "model": veh.get("model"),
                "fuel_type": veh.get("fuelType"),
                "sale_country": veh.get("saleCountry"),
                "sale_code": veh.get("saleCode"),
                "lot_number": veh.get("lot"),
            }
    return {}

def parse_subheadline(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    span = soup.select_one("span.viewlot__subheadline")
    if not span:
        return {}
    text = span.get_text()
    out = {}
    m = re.search(r"\(\s*(\d+)\s*(?:PS|CV|hp)\s*\)", text, flags=re.IGNORECASE)
    if m:
        out["cv"] = m.group(1)
    m = re.search(r"([\d\.]+)\s*km", text, flags=re.IGNORECASE)
    if m:
        out["mileage"] = m.group(1).replace(".", "")
    m = re.search(r"\b(Diesel|Gasolina|Electric(?:o)?|Hybrid(?:o)?)\b", text, flags=re.IGNORECASE)
    if m:
        out["fuel_type"] = m.group(1).capitalize()
    m = re.search(r"\b(Manual|Autom(?:á|a)tico|DSG|CVT)\b", text, flags=re.IGNORECASE)
    if m:
        out["transmission"] = m.group(1)
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        out["year"] = m.group(0)
    return out

def parse_co2(html: str) -> dict:
    m = re.search(r"(\d+)\s*(?:g/km|g\s*/\s*km)", html, flags=re.IGNORECASE)
    return {"co2": m.group(1)} if m else {}

def parse_sale_info(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    out = {}
    panel = soup.select_one("#saleInformationSidePanel .viewlot__saleinfo")
    if panel:
        name_elem = panel.select_one("h6.sale__name__subtitle")
        if name_elem:
            out["sale_name"] = name_elem.get_text(strip=True)
        flag_i = panel.select_one("div.country i.flag")
        if flag_i:
            for cls in flag_i.get("class", []):
                if cls.startswith("flag--"):
                    out["sale_info_country"] = cls.split("--")[1]
                    break
    return out

def parse_title_fallback(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    h2 = soup.select_one("h2.viewlot__headline--large")
    if not h2:
        return {}

    parts = h2.get_text(strip=True).split()
    if not parts:
        return {}

    make_tokens = []
    for i in range(1, len(parts)+1):
        candidate = " ".join(parts[:i]).upper().replace(" ", " ")  # quita nbsp
        mapped = MARCA_MAP.get(candidate)
        if mapped:
            make_tokens = parts[:i]
            make = mapped
            break
    else:
        make_tokens = [parts[0]]
        make = parts[0]

    model_tokens = parts[len(make_tokens):]
    return {
        "make": make,
        "model": " ".join(model_tokens)
    }

async def fetch_lot_html(url: str, context: BrowserContext) -> str:
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=60000)
        content = await page.content()
        table_html = await page.evaluate(
            "() => document.querySelector('table.viewlot__table')?.outerHTML"
        )
        if table_html:
            content += "\n" + table_html
        return content
    finally:
        await page.close()

async def bound_fetch(sem: asyncio.Semaphore, context: BrowserContext, url: str):
    async with sem:
        try:
            html = await fetch_lot_html(url, context)
            data = {}

            fallback = parse_title_fallback(html)
            if fallback:
                data.update(fallback)
            digital = parse_digital_data(html)
            for k, v in digital.items():
                if k == "model" and "model" in data:
                    continue
                data[k] = v
            data.update(parse_subheadline(html))
            data.update(parse_co2(html))
            data.update(parse_sale_info(html))

            m = re.search(r"<td>\s*Tipo Carrocería\s*</td>\s*<td>([^<]+)</td>", html, flags=re.IGNORECASE)
            if m:
                data["Tipo de vehículo"] = m.group(1).strip()

            table_data = parse_table(html)
            data.update(table_data)

            if "Modelo" in table_data:
                data["model"] = table_data["Modelo"]
            if "Tipo Carrocería" in table_data:
                data["Tipo de vehículo"] = table_data["Tipo Carrocería"]

            result = {}
            result.update(parse_url_params(url))
            result["sale_id"] = parse_saleid_from_bq(url)
            result["viewlot_url"] = url
            result.update(data)
            return result
        except Exception as e:
            return {"viewlot_url": url, "error": str(e)}

async def main_async(input_excel: str, storage_state: str, concurrency: int):
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    logger.info("=== INICIO scraping Fase1B ===")

    # Lee el Excel base
    df = pd.read_excel(input_excel)
    urls = df['link_ficha'].tolist()
    total = len(urls)
    logger.info(f"Comenzando scraping de {total} lotes con concurrencia={concurrency}")

    resultados_dict = {}

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=True)
        context: BrowserContext = await browser.new_context(storage_state=storage_state, locale="es-ES",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            )
        )

        # Warm-up Chromium para mejorar tiempos en cloud
        try:
            page = await context.new_page()
            await page.goto("https://www.google.com", timeout=30000)
            await page.close()
        except Exception as e:
            logger.warning(f"Warm-up Chromium falló: {e}")

        sem = asyncio.Semaphore(concurrency)
        tasks = [bound_fetch(sem, context, url) for url in urls]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            res = await coro
            completed += 1
            url = res.get("viewlot_url", "–")
            if res.get("error"):
                logger.warning(f"[{completed}/{total}] ERROR en {url}: {res['error']}")
            else:
                logger.info(f"[{completed}/{total}] Completado {url}")
            resultados_dict[url] = res

        # ----------- REINTENTOS AUTOMÁTICOS -----------
        for retry in range(1, MAX_RETRIES + 1):
            fallidas = [url for url, data in resultados_dict.items() if data.get('error')]
            if not fallidas:
                logger.info(f"Todos los lotes han sido scrapeados correctamente tras {retry-1} reintentos.")
                break
            logger.warning(f"Reintentando {len(fallidas)} lotes fallidos (intento {retry} de {MAX_RETRIES})")
            retry_tasks = [bound_fetch(sem, context, url) for url in fallidas]
            completed_retry = 0
            for coro in asyncio.as_completed(retry_tasks):
                res = await coro
                completed_retry += 1
                url = res.get("viewlot_url", "–")
                if res.get("error"):
                    logger.warning(f"[Retry {retry}] ERROR en {url}: {res['error']}")
                else:
                    logger.info(f"[Retry {retry}] Recuperado {url}")
                resultados_dict[url] = res

        await context.close()
        await browser.close()

    # --- Enriquecimiento inplace ---
    nuevas_columnas = set()
    for datos in resultados_dict.values():
        nuevas_columnas.update(set(datos.keys()))
    nuevas_columnas -= {'viewlot_url', 'error', 'lot_id', 'sale_id', 'vehicle_id'}

    for col in nuevas_columnas:
        df[col] = df['link_ficha'].map(lambda x: resultados_dict.get(x, {}).get(col))

    df['error_fase1b'] = df['link_ficha'].map(lambda x: resultados_dict.get(x, {}).get('error'))

    # Sobrescribe el mismo archivo (inplace)
    df.to_excel(input_excel, index=False)
    logger.info(f"Excel enriquecido (inplace) guardado en {input_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 1B: batch scraping Lot con Playwright y BS4 (inplace update)")
    parser.add_argument('-i','--input', required=True, help="Excel de entrada y salida (Fase1A -> enriquecido Fase1B)")
    parser.add_argument('-s','--storage', default="bca_storage_phase1.json", help="Playwright storage_state JSON")
    parser.add_argument('-c','--concurrency', type=int, default=SEM_LIMIT, help="Número de fetches paralelos")
    args = parser.parse_args()

    # --- LOGIN AUTOMÁTICO ---
    user = os.getenv("BCA_USER")
    password = os.getenv("BCA_PASS")
    if not user or not password:
        sys.exit("⚠️  Define BCA_USER y BCA_PASS en variables de entorno.")

    perform_login(
        user,
        password,
        headless=True,
        debug=False,
        screenshot_on_fail=True,
        output=pathlib.Path(args.storage)
    )

    asyncio.run(main_async(args.input, args.storage, args.concurrency))
