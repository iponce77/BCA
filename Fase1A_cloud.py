#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from selenium import webdriver
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    StaleElementReferenceException
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager

HEADLESS = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"Fase1A_Selenium_{datetime.datetime.now():%Y%m%d}.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Fase1A_Selenium")

def init_driver() -> webdriver.Chrome:
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-infobars")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(60)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                window.navigator.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            """
        }
    )

    if not HEADLESS:
        try:
            driver.maximize_window()
        except Exception:
            pass
    return driver

def filtrar_links_subasta(driver, page_num: int) -> list[str]:
    today = datetime.datetime.now()
    target = today  # 👈 Solo día en curso
    today_str = target.strftime("%d/%m")
    tipos_excluidos = {"icon--eauction", "icon--liveonline", "icon--sealed"}

    WebDriverWait(driver, 30).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    time.sleep(7)

    try:
        btn = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        btn.click()
        WebDriverWait(driver, 10).until(
            EC.invisibility_of_element_located((By.ID, "onetrust-banner-sdk"))
        )
        logger.info(f"[Calendario pág {page_num}] Cookies aceptadas")
    except Exception:
        pass

    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.listing"))
        )
    except TimeoutException:
        logger.error(f"[Calendario pág {page_num}] Timeout esperando div.listing")
        return []

    retries = 3
    while retries:
        try:
            blocks = driver.find_elements(By.CSS_SELECTOR, "div.listing")
            break
        except StaleElementReferenceException:
            retries -= 1
            time.sleep(1)

    enlaces = []
    for blk in blocks:
        try:
            fechas = [sp.text for sp in blk.find_elements(
                By.CSS_SELECTOR, "div.listing__date-info span.ng-binding"
            )]
            if not any(f.startswith("Final:") and today_str in f for f in fechas):
                continue
            clases = [i.get_attribute("class") for i in blk.find_elements(
                By.CSS_SELECTOR, "div.listing__sale-info i.icon"
            )]
            tipos = {c for cls in clases for c in cls.split() if c.startswith('icon--')}
            if tipos & tipos_excluidos:
                continue
            a = blk.find_element(By.CSS_SELECTOR, "a.listing__link.sale-title-link")
            href = a.get_attribute("href")
            if href:
                full = href if href.startswith('http') else 'https://es.bca-europe.com' + href
                enlaces.append(full)
        except StaleElementReferenceException:
            logger.warning(f"[Calendario pág {page_num}] stale element bloque omitido")
        except Exception as e:
            logger.warning(f"[Calendario pág {page_num}] Error filtrando: {e}")

    logger.info(f"[Calendario pág {page_num}] captadas {len(enlaces)} subastas válidas")
    return enlaces

def scrape_lot_links_selenium(driver: webdriver.Chrome, sub_url: str) -> set[str]:
    link_set: set[str] = set()
    try:
        driver.get(sub_url)
    except TimeoutException:
        logger.error(f"[B] Timeout al abrir subasta: {sub_url}")
        return link_set

    time.sleep(7)
    try:
        btn = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        btn.click()
        WebDriverWait(driver, 10).until(
            EC.invisibility_of_element_located((By.ID, "onetrust-banner-sdk"))
        )
    except Exception:
        pass

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/Lot?id=']"))
        )
    except TimeoutException:
        return link_set

    page_num = 1
    while True:
        time.sleep(3)
        elems = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/Lot?id="]')
        for e in elems:
            try:
                href = e.get_attribute("href")
                if href:
                    link_set.add(href)
            except StaleElementReferenceException:
                logger.warning(f"[Page {page_num}] Stale element en {sub_url}, link omitido")
                continue
        try:
            next_btn = driver.find_element(By.ID, "nextPage")
            if not next_btn.is_enabled() or "disabled" in (next_btn.get_attribute("class") or ""):
                break
            next_btn.click()
            page_num += 1
        except Exception:
            break

    return link_set

def worker(sub_urls: list[str]) -> list[str]:
    drv = init_driver()
    resultados: list[str] = []
    try:
        for url in sub_urls:
            try:
                lotes = scrape_lot_links_selenium(drv, url)
                resultados.extend(lotes)
            except Exception as e:
                logger.error(f"Error scraping subasta {url}: {e}", exc_info=True)
                continue
    finally:
        drv.quit()
    return resultados

def main():
    calendar_url = "https://es.bca-europe.com/buyer/facetedSearch/saleCalendar?bq=&cultureCode=en"
    driver = init_driver()
    driver.get(calendar_url)

    links = filtrar_links_subasta(driver, page_num=1)

    pag_links = driver.find_elements(By.CSS_SELECTOR,
                                     "nav.nav--pagination a.nav__link--pagination")
    pages = sorted({el.text.strip() for el in pag_links if el.text.strip().isdigit()}, key=int)
    for num in pages[1:]:
        try:
            btn = driver.find_element(
                By.XPATH,
                f"//nav[contains(@class,'nav--pagination')]//a[normalize-space()='{num}']"
            )
            btn.click()
            time.sleep(2)
            links.extend(filtrar_links_subasta(driver, page_num=int(num)))
        except Exception as e:
            logger.warning(f"[Calendario pág {num}] Error paginando: {e}")

    driver.quit()
    unique_sub = list(dict.fromkeys(links))
    logger.info(f"Total subastas extraídas: {len(unique_sub)}")

    if not unique_sub:
        logger.info("⛔ No hay subastas activas para el día en curso. No se genera Excel.")
        return

    n_workers = 4
    chunks = [unique_sub[i::n_workers] for i in range(n_workers)]
    all_lotes: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = [exe.submit(worker, chunk) for chunk in chunks]
        for f in as_completed(futures):
            try:
                all_lotes.extend(f.result())
            except Exception as e:
                logger.error(f"Error en worker: {e}", exc_info=True)

    unique_lotes = list(dict.fromkeys(all_lotes))
    if not unique_lotes:
        logger.info("⛔ No se encontraron lotes en las subastas del día. No se genera Excel.")
        return

    df = pd.DataFrame({"link_ficha": unique_lotes})
    out_name = f"fichas_vehiculos_{datetime.datetime.now():%Y%m%d}.xlsx"
    df.to_excel(out_name, index=False)
    logger.info(f"✅ Guardados {len(unique_lotes)} fichas en {out_name}")

if __name__ == "__main__":
    main()
