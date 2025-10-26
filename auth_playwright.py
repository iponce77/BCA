# auth_playwright.py
# Versión limpia — no hace login por sí misma

import json
import pathlib
import requests

def get_session(storage_path: str | pathlib.Path = "bca_storage.json") -> requests.Session:
    """
    Crea y devuelve un requests.Session con cookies cargadas desde storage_path si existe.
    No intenta hacer login automático: si no hay cookies, devuelve Session limpia.
    """
    storage_path = pathlib.Path(storage_path)
    session = requests.Session()

    if storage_path.exists():
        try:
            state = json.loads(storage_path.read_text(encoding="utf-8"))
            for ck in state.get("cookies", []):
                dom = ck.get("domain", "").lstrip(".")
                session.cookies.set(ck["name"], ck["value"], domain=dom, path=ck.get("path", "/"))
                if dom.endswith("bca.com"):
                    # Añadimos alias para dominio específico ES si aplica
                    session.cookies.set(ck["name"], ck["value"], domain="es.bca-europe.com", path=ck.get("path", "/"))
            print(f"✅ Cookies cargadas desde {storage_path}")
        except Exception as e:
            print(f"⚠️ No se pudieron cargar cookies desde {storage_path}: {e}")
    else:
        print(f"ℹ️ No existe storage_state: {storage_path} (Session sin cookies)")

    return session

