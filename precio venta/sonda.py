#!/usr/bin/env python3
import os, sys, json, requests

EMPRESA = int(os.getenv("GANVAM_EMPRESA", "60"))
BASE = os.getenv("GANVAM_BASE", "https://webapi.ganvam.es/api/AutomocionTarifaVOVentas")
TOKEN = os.getenv("GANVAM_TOKEN")
STATE_FILE = os.getenv("GANVAM_STATE_FILE", "ganvam_state.json")
PROBE_N = int(os.getenv("GANVAM_PROBE_N", "3"))        # cuántos periodos hacia delante probar
START_FALLBACK = int(os.getenv("GANVAM_PERIODO_START", "212"))  # si no hay state file

if not TOKEN:
    print("⛔ Falta GANVAM_TOKEN", file=sys.stderr)
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def get_json(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"[error] {url} → {e}", file=sys.stderr)
        return None

def load_last_period() -> int:
    if not os.path.exists(STATE_FILE):
        print(f"[info] No existe {STATE_FILE}, usando fallback {START_FALLBACK}")
        return START_FALLBACK
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        lp = int(data.get("last_period", START_FALLBACK))
        return lp
    except Exception:
        return START_FALLBACK

def detect_new_period(current: int, probe_n: int) -> int | None:
    for delta in range(1, probe_n + 1):
        cand = current + delta
        url = f"{BASE}/marcas/{EMPRESA}/{cand}"
        data = get_json(url)
        if isinstance(data, list) and len(data) > 0:
            return cand
    return None

def save_last_period(period: int):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_period": period}, f, ensure_ascii=False, indent=2)

def main():
    last_period = load_last_period()
    nuevo = detect_new_period(last_period, PROBE_N)

    if nuevo:
        print(f"[ok] Detectado nuevo periodo: {nuevo}")
        # Actualizamos el estado en el workspace; el YAML hará commit/push.
        save_last_period(nuevo)
        # Outputs para Actions (compatible con set-output “antiguo” y env GITHUB_OUTPUT)
        print(f"::set-output name=nuevo_periodo::true")
        print(f"::set-output name=periodo::{nuevo}")
        # Soporte moderno: escribir a GITHUB_OUTPUT si existe
        gh_out = os.getenv("GITHUB_OUTPUT")
        if gh_out:
            with open(gh_out, "a") as gho:
                gho.write(f"nuevo_periodo=true\nperiodo={nuevo}\n")
        sys.exit(0)
    else:
        print(f"[ok] Sin cambios. Periodo vigente = {last_period}")
        print(f"::set-output name=nuevo_periodo::false")
        print(f"::set-output name=periodo::{last_period}")
        gh_out = os.getenv("GITHUB_OUTPUT")
        if gh_out:
            with open(gh_out, "a") as gho:
                gho.write(f"nuevo_periodo=false\nperiodo={last_period}\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
