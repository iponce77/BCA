#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
login_poc.py – Login en BCA Europe con Playwright (sync), manejo de OneTrust,
persistencia de cookies y compatibilidad headless para GitHub Actions.

- Firma compatible: perform_login(user, password, headless, debug, screenshot_on_fail, output)
- Esperas robustas (visible + click + fill)
- Visita /buyer para “sellar” cookies del subdominio
- Reintento automático si detecta “sesión caducada”
"""

import json
import os
import pathlib
import sys
import time
from datetime import datetime
from typing import List

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

LOGIN_URL = "https://es.bca-europe.com/login"
BUYER_URL = "https://es.bca-europe.com/buyer"

COOKIE_ACCEPT_SELECTORS: List[str] = [
    "#onetrust-reject-all-handler",
    "#onetrust-accept-btn-handler",
]

EMAIL_SELECTORS: List[str] = [
    "input#username",
    "input[name='username']",
    "input[type='email']",
]

PASSWORD_SELECTORS: List[str] = [
    "input#password",
    "input[name='password']",
    "input[type='password']",
]

BUTTON_SELECTORS: List[str] = [
    "button#loginButton",
    "button[type='submit']",
    "button:has-text('Log in')",
    "button:has-text('Iniciar sesión')",
]

SUCCESS_COOKIES = {"XSRF-TOKEN", "idsrv"}   # indicadores de sesión iniciada
DEFAULT_TIMEOUT = 30_000  # ms


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, flush=True)


def _click_if_exists(page, selectors, timeout=5_000) -> bool:
    """Espera visibilidad y hace click si el selector existe."""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count():
                loc.wait_for(state="visible", timeout=timeout)
                loc.click()
                return True
        except Exception:
            continue
    return False


def _find_and_fill(page, selectors, value, timeout=15_000) -> bool:
    """Espera visibilidad, hace click y rellena el input."""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.count():
                loc.wait_for(state="visible", timeout=timeout)
                loc.click()
                loc.fill(value)
                return True
        except Exception:
            continue
    return False


def _handle_cookies_banner(page) -> None:
    """Cierra el banner OneTrust si aparece (no bloqueante)."""
    try:
        _click_if_exists(page, COOKIE_ACCEPT_SELECTORS, timeout=3_000)
    except Exception:
        pass


def _has_success_cookies(context) -> bool:
    try:
        names = {c.get("name", "") for c in context.cookies()}
        return bool(SUCCESS_COOKIES & names)
    except Exception:
        return False


def perform_login(user: str, password: str, headless: bool, debug: bool, screenshot_on_fail: bool, output: pathlib.Path):
    """Login robusto, guarda storage_state en `output` y sale con sys.exit en caso de fallo."""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--disable-dev-shm-usage",
                "--window-size=1920,1080",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = browser.new_context(
            locale="es-ES",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        # Anti-detección básica
        context.add_init_script("Object.defineProperty(navigator,'webdriver',{get:() => undefined});")
        context.set_default_timeout(DEFAULT_TIMEOUT)
        context.set_default_navigation_timeout(DEFAULT_TIMEOUT)

        page = context.new_page()

        def do_login_once() -> bool:
            _log(debug, "[login] Abriendo página de login…")
            page.goto(LOGIN_URL, timeout=45_000, wait_until="load")
            _handle_cookies_banner(page)

            _log(debug, "[login] Rellenando usuario…")
            if not _find_and_fill(page, EMAIL_SELECTORS, user, timeout=15_000):
                return False

            _log(debug, "[login] Rellenando contraseña…")
            if not _find_and_fill(page, PASSWORD_SELECTORS, password, timeout=15_000):
                return False

            _log(debug, "[login] Pulsando botón de acceso…")
            if not _click_if_exists(page, BUTTON_SELECTORS, timeout=10_000):
                return False

            try:
                page.wait_for_load_state("networkidle", timeout=20_000)
            except PWTimeout:
                pass

            _handle_cookies_banner(page)

            # Visita BUYER para “sellar” cookies del subdominio
            try:
                _log(debug, "[login] Visitando /buyer para validar sesión…")
                page.goto(BUYER_URL, timeout=20_000, wait_until="load")
                page.wait_for_load_state("networkidle", timeout=15_000)
            except PWTimeout:
                _log(debug, "⚠️ Timeout visitando /buyer (continuo)")

            ok = _has_success_cookies(context)
            _log(debug, f"[login] Cookies de éxito: {'OK' if ok else 'NO'}")
            return ok

        ok = do_login_once()

        # Reintento si detecta texto de “sesión caducada”
        if not ok:
            try:
                body = page.content().lower()
            except Exception:
                body = ""
            if any(k in body for k in ("sesión ha caducado", "session has expired", "session expired")):
                _log(True, "⚠️ Detectada posible 'sesión caducada'. Reintentando login desde cero…")
                try:
                    context.clear_cookies()
                except Exception:
                    pass
                try:
                    page.close()
                except Exception:
                    pass
                page = context.new_page()
                ok = do_login_once()

        if not ok:
            if screenshot_on_fail:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    page.screenshot(path=f"login_fail_{ts}.png", full_page=True)
                    print(f"📸 Captura guardada en login_fail_{ts}.png")
                except Exception:
                    pass
            browser.close()
            sys.exit("⏰ No se detectaron cookies de sesión. Revisa credenciales/cambios en el login.")

        # Guardar storage
        state = context.storage_state()
        output.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        if debug:
            print(f"✅ Login OK. {len(state.get('cookies', []))} cookies guardadas en {output}")
        browser.close()

