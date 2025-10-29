# scripts/dgt/parse_dgt.py
from __future__ import annotations
import re, os
from datetime import datetime
from pathlib import Path
import pandas as pd

# ---- Tablas y utilidades (copiadas/adaptadas de tu procesar_dgt_final.py) ----
combustible_map = {
    "0":"GASOLINA","1":"DIESEL","2":"ELECTRICO","3":"OTRO","4":"BUTANO","5":"SOLAR","6":"GLP",
    "7":"GNC","8":"GNL","9":"HIDROGENO","A":"BIOMETANO","B":"ETANOL","C":"BIODIESEL"
}

tipo_vehiculo_map = {
    "01":"CAMIÓN","02":"FURGONETA","03":"FURGÓN","04":"DERIVADO TURISMO","05":"TRACTOCAMIÓN",
    "10":"AUTOCARAVANA","20":"AUTOBÚS","30":"TODO TERRENO","31":"AUTOBÚS","40":"TURISMO",
    "41":"CUADRICICLO","50":"MOTOCICLETA DE 2 RUEDAS","51":"MOTOCICLETA CON SIDECAR",
    "52":"TRICICLO","60":"VEHÍCULO MIXTO ADAPTABLE","70":"VEHÍCULO ESPECIAL","80":"QUAD",
    "90":"CICLOMOTOR DE 2 RUEDAS","91":"CICLOMOTOR CON SIDECAR","92":"CICLOMOTOR DE 3 RUEDAS",
    "R1":"REMOLQUE PLATAFORMA","R2":"REMOLQUE CISTERNA","R3":"REMOLQUE CAJA ABIERTA","R4":"REMOLQUE CAJA CERRADA",
    "S1":"SEMIRREMOLQUE PLATAFORMA","S2":"SEMIRREMOLQUE CISTERNA","S3":"SEMIRREMOLQUE CAJA ABIERTA","S4":"SEMIRREMOLQUE CAJA CERRADA"
}

codigo_ine_to_provincia = {
    "01":"ÁLAVA","02":"ALBACETE","03":"ALICANTE","04":"ALMERÍA","05":"ÁVILA","06":"BADAJOZ","07":"BALEARES",
    "08":"BARCELONA","09":"BURGOS","10":"CÁCERES","11":"CÁDIZ","12":"CASTELLÓN","13":"CIUDAD REAL","14":"CÓRDOBA",
    "15":"A CORUÑA","16":"CUENCA","17":"GIRONA","18":"GRANADA","19":"GUADALAJARA","20":"GUIPÚZCOA","21":"HUELVA",
    "22":"HUESCA","23":"JAÉN","24":"LEÓN","25":"LLEIDA","26":"LA RIOJA","27":"LUGO","28":"MADRID","29":"MÁLAGA",
    "30":"MURCIA","31":"NAVARRA","32":"OURENSE","33":"ASTURIAS","34":"PALENCIA","35":"LAS PALMAS","36":"PONTEVEDRA",
    "37":"SALAMANCA","38":"SANTA CRUZ DE TENERIFE","39":"CANTABRIA","40":"SEGOVIA","41":"SEVILLA","42":"SORIA",
    "43":"TARRAGONA","44":"TERUEL","45":"TOLEDO","46":"VALENCIA","47":"VALLADOLID","48":"VIZCAYA","49":"ZAMORA",
    "50":"ZARAGOZA","51":"CEUTA","52":"MELILLA"
}

def detectar_version(nombre_archivo: str) -> str:
    return "post_2025" if re.search(r"20(2[5-9]|[3-9]\d)", nombre_archivo) else "pre_2025"

def extraer_transmision(nombre_archivo: str) -> str | None:
    m = re.search(r"(20\d{2})(\d{2})", nombre_archivo)
    return f"{m.group(1)}-{m.group(2)}" if m else None

def _procesar_linea(line: str, version: str) -> dict:
    # (idéntica a tu lógica, compactada)
    try:
        fecha_raw = line[:8]
        fecha_mat = datetime.strptime(fecha_raw, "%d%m%Y").strftime("%d/%m/%Y")
    except Exception:
        fecha_mat = None

    vin, combustible, vin_pos = None, None, None
    if version == "post_2025":
        m = re.search(r'([A-Z0-9]{11})(\*{11,})', line)
        if m:
            vin = m.group(1) + m.group(2)
            vin_end = m.end()
            bloque_comb = line[vin_end:vin_end+3]
            if len(bloque_comb) == 3 and bloque_comb[0].isdigit() and bloque_comb[2].isdigit():
                combustible = combustible_map.get(bloque_comb[2])
            vin_pos = m.start()
    else:
        m_alpha = re.search(r'[A-Z]', line)
        if m_alpha:
            search_area = line[m_alpha.start():]
            m = re.search(r'([0-9][A-Z0-9]{17})(?=\s+[0-9][A-Z0-9][0-9])', search_area)
            if m:
                vin = m.group(1)
                vin_pos = m_alpha.start() + m.start()
                bloque = re.search(r'\s+([0-9][A-Z0-9][0-9])', search_area[m.end():])
                if bloque:
                    combustible = combustible_map.get(bloque.group(1)[2])

    m_alpha = re.search(r'[A-Z]', line)
    if not m_alpha or vin_pos is None:
        return {}

    start_idx = m_alpha.start()
    bloque = line[start_idx:vin_pos].strip()
    m_marca = re.match(r'([A-Z][A-Z0-9&\-]*)', bloque)
    marca = m_marca.group(1) if m_marca else None
    if marca == "MERCEDES": marca = "MERCEDES-BENZ"
    elif marca == "LAND" and "ROVER" in bloque: marca = "LAND ROVER"

    texto_sin_marca = bloque[len(marca):].strip() if marca else bloque
    palabras = [p for p in texto_sin_marca.split() if not (marca and p.upper() == marca.upper())]
    if palabras:
        modelo = palabras[0] + palabras[1] if len(palabras) >= 2 and len(palabras[0]) == 1 and palabras[1][0].isdigit() else palabras[0]
        submodelo = " ".join(palabras)
    else:
        modelo, submodelo = "", ""

    ine = re.search(r'(B0\d{6}|A0\d{6}|B18\d{5}|A18\d{5}|B22\d{5})([A-Z].{3,})', line)
    codigo_ine = ine.group(1)[-5:] if ine else None
    localidad = ine.group(2).split()[0] if ine else None
    provincia = codigo_ine_to_provincia.get(codigo_ine[:2]) if codigo_ine else None

    tipo_vehiculo = None
    if vin:
        post_vin = re.sub(r"[^A-Z0-9]", "", line[vin_pos + len(vin):])
        tipo_vehiculo = tipo_vehiculo_map.get(post_vin[:2], "DESCONOCIDO") if len(post_vin) >= 2 else "DESCONOCIDO"

    return {
        "fecha_mat": fecha_mat, "marca": marca, "modelo": modelo, "submodelo": submodelo,
        "vin": vin, "combustible": combustible, "codigo_ine": codigo_ine, "localidad": localidad,
        "provincia": provincia, "tipo_vehiculo": tipo_vehiculo
    }

def txt_to_parquet(txt_path: Path, out_parquet: Path) -> None:
    archivo = txt_path.name
    version = detectar_version(archivo)
    transmision = extraer_transmision(archivo)

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lineas = [ln for ln in f if ln.strip()]

    df = pd.DataFrame([d for d in (_procesar_linea(ln, version) for ln in lineas) if d])
    if df.empty:
        # escribimos parquet vacío con esquema mínimo
        pd.DataFrame(columns=["fecha_mat","marca","modelo","submodelo","vin","combustible",
                              "codigo_ine","localidad","provincia","tipo_vehiculo","transmision",
                              "año_mat","mes_mat","antiguedad_anios","nombre_archivo",
                              "es_cruzable","codigo_provincia"]).to_parquet(out_parquet, index=False)
        return

    df["transmision"] = transmision
    fechas = pd.to_datetime(df["fecha_mat"], errors="coerce", dayfirst=True)
    df["año_mat"] = fechas.dt.year
    df["mes_mat"] = fechas.dt.month
    df["antiguedad_anios"] = datetime.now().year - fechas.dt.year
    df["nombre_archivo"] = archivo
    df["es_cruzable"] = df["modelo"].notna() & df["codigo_ine"].notna()
    df["codigo_provincia"] = df["codigo_ine"].str[:2]

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
