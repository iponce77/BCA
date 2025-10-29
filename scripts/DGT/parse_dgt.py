# scripts/DGT/parse_dgt.py
from __future__ import annotations
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# =========================
#   MAPEOS Y CONSTANTES
# =========================
combustible_map: Dict[str, str] = {
    "0": "GASOLINA", "1": "DIESEL", "2": "ELECTRICO", "3": "OTRO", "4": "BUTANO",
    "5": "SOLAR", "6": "GLP", "7": "GNC", "8": "GNL", "9": "HIDROGENO",
    "A": "BIOMETANO", "B": "ETANOL", "C": "BIODIESEL",
}

tipo_vehiculo_map: Dict[str, str] = {
    "01": "CAMIÓN", "02": "FURGONETA", "03": "FURGÓN", "04": "DERIVADO TURISMO",
    "05": "TRACTOCAMIÓN", "10": "AUTOCARAVANA", "20": "AUTOBÚS", "30": "TODO TERRENO",
    "31": "AUTOBÚS", "40": "TURISMO", "41": "CUADRICICLO", "50": "MOTOCICLETA DE 2 RUEDAS",
    "51": "MOTOCICLETA CON SIDECAR", "52": "TRICICLO", "60": "VEHÍCULO MIXTO ADAPTABLE",
    "70": "VEHÍCULO ESPECIAL", "80": "QUAD", "90": "CICLOMOTOR DE 2 RUEDAS",
    "91": "CICLOMOTOR CON SIDECAR", "92": "CICLOMOTOR DE 3 RUEDAS",
    "R1": "REMOLQUE PLATAFORMA", "R2": "REMOLQUE CISTERNA", "R3": "REMOLQUE CAJA ABIERTA",
    "R4": "REMOLQUE CAJA CERRADA", "S1": "SEMIRREMOLQUE PLATAFORMA",
    "S2": "SEMIRREMOLQUE CISTERNA", "S3": "SEMIRREMOLQUE CAJA ABIERTA",
    "S4": "SEMIRREMOLQUE CAJA CERRADA",
}

codigo_ine_to_provincia: Dict[str, str] = {
    "01": "ÁLAVA", "02": "ALBACETE", "03": "ALICANTE", "04": "ALMERÍA", "05": "ÁVILA",
    "06": "BADAJOZ", "07": "BALEARES", "08": "BARCELONA", "09": "BURGOS", "10": "CÁCERES",
    "11": "CÁDIZ", "12": "CASTELLÓN", "13": "CIUDAD REAL", "14": "CÓRDOBA", "15": "A CORUÑA",
    "16": "CUENCA", "17": "GIRONA", "18": "GRANADA", "19": "GUADALAJARA", "20": "GUIPÚZCOA",
    "21": "HUELVA", "22": "HUESCA", "23": "JAÉN", "24": "LEÓN", "25": "LLEIDA",
    "26": "LA RIOJA", "27": "LUGO", "28": "MADRID", "29": "MÁLAGA", "30": "MURCIA",
    "31": "NAVARRA", "32": "OURENSE", "33": "ASTURIAS", "34": "PALENCIA",
    "35": "LAS PALMAS", "36": "PONTEVEDRA", "37": "SALAMANCA",
    "38": "SANTA CRUZ DE TENERIFE", "39": "CANTABRIA", "40": "SEGOVIA", "41": "SEVILLA",
    "42": "SORIA", "43": "TARRAGONA", "44": "TERUEL", "45": "TOLEDO", "46": "VALENCIA",
    "47": "VALLADOLID", "48": "VIZCAYA", "49": "ZAMORA", "50": "ZARAGOZA",
    "51": "CEUTA", "52": "MELILLA",
}

# =========================
#   REGEX COMPILADOS
# =========================
RX_ANY_ALPHA = re.compile(r"[A-Z]")
RX_VIN_POST25 = re.compile(r"([A-Z0-9]{11})(\*{11,})")
RX_VIN_PRE25_AFTER_ALPHA = re.compile(r"([0-9][A-Z0-9]{17})(?=\s+[0-9][A-Z0-9][0-9])")
RX_BLOQUE_PRE25 = re.compile(r"\s+([0-9][A-Z0-9][0-9])")
RX_AAAA_MM_IN_NAME = re.compile(r"(20\d{2})(\d{2})")
RX_INE_BLOQUE = re.compile(r"(B0\d{6}|A0\d{6}|B18\d{5}|A18\d{5}|B22\d{5})([A-Z].{3,})")


# =========================
#   FUNCIONES AUXILIARES
# =========================
def detectar_version(nombre_archivo: str) -> str:
    # Heurística simple: si el nombre contiene año >= 2025 -> post_2025
    m = RX_AAAA_MM_IN_NAME.search(nombre_archivo)
    if m and int(m.group(1)) >= 2025:
        return "post_2025"
    return "pre_2025"

def extraer_transmision(nombre_archivo: str) -> Optional[str]:
    m = RX_AAAA_MM_IN_NAME.search(nombre_archivo)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def _procesar_linea(line: str, version: str) -> Dict[str, Optional[str]]:
    """Extrae campos de una línea. Devuelve {} si no se puede parsear."""
    # fecha de matriculación (ddmmyyyy -> dd/mm/yyyy)
    try:
        fecha_raw = line[:8]
        fecha_mat = datetime.strptime(fecha_raw, "%d%m%Y").strftime("%d/%m/%Y")
    except Exception:
        fecha_mat = None

    vin, combustible, vin_pos = None, None, None

    if version == "post_2025":
        m = RX_VIN_POST25.search(line)
        if m:
            vin = m.group(1) + m.group(2)
            vin_end = m.end()
            bloque_comb = line[vin_end : vin_end + 3]
            if (
                len(bloque_comb) == 3
                and bloque_comb[0].isdigit()
                and bloque_comb[2].isdigit()
            ):
                combustible = combustible_map.get(bloque_comb[2])
            vin_pos = m.start()
    else:
        m_alpha = RX_ANY_ALPHA.search(line)
        if m_alpha:
            search_area = line[m_alpha.start() :]
            m = RX_VIN_PRE25_AFTER_ALPHA.search(search_area)
            if m:
                vin = m.group(1)
                vin_pos = m_alpha.start() + m.start()
                bloque = RX_BLOQUE_PRE25.search(search_area[m.end() :])
                if bloque:
                    combustible = combustible_map.get(bloque.group(1)[2])

    m_alpha = RX_ANY_ALPHA.search(line)
    if not m_alpha or vin_pos is None:
        return {}

    # Marca / modelo / submodelo
    start_idx = m_alpha.start()
    bloque = line[start_idx:vin_pos].strip()
    m_marca = re.match(r"([A-Z][A-Z0-9&\-]*)", bloque)
    marca = m_marca.group(1) if m_marca else None
    if marca == "MERCEDES":
        marca = "MERCEDES-BENZ"
    elif marca == "LAND" and "ROVER" in bloque:
        marca = "LAND ROVER"

    texto_sin_marca = bloque[len(marca) :].strip() if marca else bloque
    palabras = [p for p in texto_sin_marca.split() if not (marca and p.upper() == marca.upper())]
    if palabras:
        if len(palabras) >= 2 and len(palabras[0]) == 1 and palabras[1][0].isdigit():
            modelo = palabras[0] + palabras[1]
        else:
            modelo = palabras[0]
        submodelo = " ".join(palabras)
    else:
        modelo, submodelo = "", ""

    # INE / localidad / provincia
    ine = RX_INE_BLOQUE.search(line)
    codigo_ine = ine.group(1)[-5:] if ine else None
    localidad = ine.group(2).split()[0] if ine else None
    provincia = codigo_ine_to_provincia.get(codigo_ine[:2]) if codigo_ine else None

    # Tipo de vehículo (basado en posiciones post-VIN)
    tipo_vehiculo = None
    if vin:
        post_vin = re.sub(r"[^A-Z0-9]", "", line[vin_pos + len(vin) :])
        tipo_vehiculo = (
            tipo_vehiculo_map.get(post_vin[:2], "DESCONOCIDO") if len(post_vin) >= 2 else "DESCONOCIDO"
        )

    return {
        "fecha_mat": fecha_mat,
        "marca": marca,
        "modelo": modelo,
        "submodelo": submodelo,
        "vin": vin,
        "combustible": combustible,
        "codigo_ine": codigo_ine,
        "localidad": localidad,
        "provincia": provincia,
        "tipo_vehiculo": tipo_vehiculo,
    }


# =========================
#   TXT -> PARQUET (stream)
# =========================
def _iter_line_dicts(txt_path: Path, version: str, progress_every: int = 50_000) -> Iterable[Dict]:
    """Generador: produce dicts parseados línea a línea (streaming)."""
    total_seen = 0
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            d = _procesar_linea(line, version)
            if d:
                yield d
            total_seen += 1
            if total_seen % progress_every == 0:
                logger.info(f"Parseando {txt_path.name}: {total_seen} líneas...")

def _finalize_batch(df: pd.DataFrame, nombre_archivo: str) -> pd.DataFrame:
    """Añade columnas derivadas al lote (transmisión/fecha/flags)."""
    transmision = extraer_transmision(nombre_archivo)
    df["transmision"] = transmision

    fechas = pd.to_datetime(df["fecha_mat"], errors="coerce", dayfirst=True)
    df["año_mat"] = fechas.dt.year
    df["mes_mat"] = fechas.dt.month
    now_year = datetime.now().year
    df["antiguedad_anios"] = now_year - fechas.dt.year
    df["nombre_archivo"] = nombre_archivo
    df["es_cruzable"] = df["modelo"].notna() & df["codigo_ine"].notna()
    df["codigo_provincia"] = df["codigo_ine"].str[:2]
    return df


def txt_to_parquet(txt_path: Path, out_parquet: Path, chunk_rows: int = 50_000) -> None:
    """
    Convierte el .txt a Parquet con PyArrow en streaming (por bloques).
    - Lee línea a línea (no carga todo en memoria).
    - Aplica regex compilados.
    - Escribe Parquet con ParquetWriter (schema del primer bloque).
    """
    logger.info(f"TXT→Parquet | {txt_path.name} -> {out_parquet.name}")
    version = detectar_version(txt_path.name)

    writer: Optional[pq.ParquetWriter] = None
    rows_written = 0
    batch: List[Dict] = []

    try:
        for d in _iter_line_dicts(txt_path, version):
            batch.append(d)
            if len(batch) >= chunk_rows:
                df = pd.DataFrame(batch)
                if not df.empty:
                    df = _finalize_batch(df, txt_path.name)
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if writer is None:
                        out_parquet.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(
                            out_parquet.as_posix(), table.schema, compression="snappy"
                        )
                    writer.write_table(table)
                    rows_written += len(df)
                batch.clear()

        # Último lote
        if batch:
            df = pd.DataFrame(batch)
            if not df.empty:
                df = _finalize_batch(df, txt_path.name)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    out_parquet.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(
                        out_parquet.as_posix(), table.schema, compression="snappy"
                    )
                writer.write_table(table)
                rows_written += len(df)
            batch.clear()

        # Si no se escribió nada, crear un Parquet vacío con esquema mínimo
        if writer is None and rows_written == 0:
            logger.warning(f"Archivo sin filas parseables: {txt_path.name}. Escribiendo parquet vacío.")
            cols = [
                "fecha_mat","marca","modelo","submodelo","vin","combustible",
                "codigo_ine","localidad","provincia","tipo_vehiculo","transmision",
                "año_mat","mes_mat","antiguedad_anios","nombre_archivo",
                "es_cruzable","codigo_provincia"
            ]
            pd.DataFrame(columns=cols).to_parquet(out_parquet, index=False)

    finally:
        if writer is not None:
            writer.close()
            logger.info(f"Parquet escrito: {out_parquet.name} | filas ~{rows_written}")
        else:
            logger.info(f"Parquet escrito (vacío): {out_parquet.name}")
