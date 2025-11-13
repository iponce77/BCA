# MERGE_ONLY.py - sin logging
# Uso: Ejecuta tras el scraping. Debe existir "_output_excels/" con los exports
# y un Excel de fallback (fichas_vehiculos_YYYYMMDD.xlsx) con columna 'link_ficha'.

import zipfile, io, os, pandas as pd, re, shutil
from pathlib import Path
from datetime import datetime

def canonicalize_lot_url(u: str) -> str:
    u = (u or "").strip()
    if not u: return u
    u = re.sub(r"#.*$", "", u)
    if "?" in u:
        base, q = u.split("?", 1)
        parts = [p for p in q.split("&") if p]
        kv = [p.split("=", 1) if "=" in p else [p, ""] for p in parts]
        kv = [(k.lower(), v) for k, v in kv]; kv.sort()
        qn = "&".join([f"{k}={v}" if v else k for k, v in kv])
        u = f"{base}?{qn}"
    return u

def try_read_export(path_or_bytes):
    meta = {"Nombre Subasta": None, "Fecha Subasta": None}
    try:
        df_none = pd.read_excel(path_or_bytes, header=None)
        raw = df_none.copy()
    except Exception:
        try:
            if isinstance(path_or_bytes, (str, Path)):
                with open(path_or_bytes, "rb") as f:
                    data = f.read()
            else:
                data = path_or_bytes.read() if hasattr(path_or_bytes, "read") else path_or_bytes
            tables = pd.read_html(io.BytesIO(data))
            raw = max(tables, key=lambda t: t.shape[0]) if tables else pd.DataFrame()
        except Exception:
            return pd.DataFrame(), meta
    if raw.empty: return pd.DataFrame(), meta
    try:
        df_std = pd.read_excel(path_or_bytes, header=6)
        raw2 = pd.read_excel(path_or_bytes, header=None)
        try:
            if str(raw2.iat[1, 1]).strip().lower().startswith("nombre de subasta"):
                meta["Nombre Subasta"] = str(raw2.iat[1, 2]).strip() if raw2.shape[1] > 2 else None
        except Exception: pass
        try:
            if str(raw2.iat[4, 4]).strip().lower().startswith("fecha"):
                meta["Fecha Subasta"] = str(raw2.iat[4, 5]).strip()
        except Exception: pass
        return df_std, meta
    except Exception:
        pass
    header_row = None
    for r in range(min(15, len(raw))):
        row_vals = [str(x).strip().lower() for x in list(raw.iloc[r].values)]
        if any(k in v for v in row_vals for k in ["nº lote","nºlote","numero de lote"]):
            header_row = r; break
    if header_row is not None:
        try:
            df2 = pd.read_excel(path_or_bytes, header=header_row)
            return df2, meta
        except Exception:
            pass
    return raw, meta

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    new_cols = []
    for c in df.columns:
        s = str(c).strip(); s_low = s.lower()
        if re.search(r"(ver.*vínculo.*lote)|(vinculo.*lote)|(^url$)|(^link$)|enlace", s_low):
            new_cols.append("Ver vínculo del lote")
        elif s_low in {"nº lote", "nºlote", "no lote", "n\\u00ba lote"}:
            new_cols.append("Nº Lote")
        else:
            new_cols.append(s)
    df = df.copy(); df.columns = new_cols
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    for c in drop_cols:
        if df[c].isna().all():
            df = df.drop(columns=[c])
    if "Ver vínculo del lote" in df.columns:
        df["Ver vínculo del lote"] = df["Ver vínculo del lote"].astype(str).str.strip()
    return df

def _collect_export_paths(exports_source: str) -> list[Path]:
    """
    exports_source puede ser una CARPETA (p.ej. '_output_excels') o un ZIP ('_output_excels.zip').
    Devuelve rutas a todos los .xls/.xlsx encontrados.
    """
    p = Path(exports_source)
    paths = []
    if p.exists() and p.is_dir():
        for f in sorted(p.rglob("*")):
            if re.search(r"\.(xls|xlsx)$", f.name, re.I):
                paths.append(f)
        return paths
    # Si es archivo y termina en .zip, lo trato como zip
    if p.exists() and p.is_file() and p.suffix.lower() == ".zip":
        tmpdir = Path("_tmp_exports")
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(tmpdir)
        for f in sorted(tmpdir.rglob("*")):
            if re.search(r"\.(xls|xlsx)$", f.name, re.I):
                paths.append(f)
        return paths
    raise FileNotFoundError(f"No encuentro carpeta/zip válido: {exports_source}")

def run_merge(exports_source: str, fallback_path: str, out_path: str):
    # 1) Reunir ficheros desde CARPETA o ZIP
    export_paths = _collect_export_paths(exports_source)
    frames = []
    for path in export_paths:
        df, meta = try_read_export(str(path))
        if df is None or df.empty: continue
        df = standardize_columns(df)
        keep = pd.Series(True, index=df.index)
        if "Ver vínculo del lote" in df.columns:
            keep &= df["Ver vínculo del lote"].astype(str).str.strip().ne("")
        if "Nº Lote" in df.columns:
            keep &= df["Nº Lote"].astype(str).str.strip().ne("")
        df = df[keep].copy()
        df.insert(0, "Nombre Subasta", meta.get("Nombre Subasta"))
        df.insert(1, "Fecha Subasta", meta.get("Fecha Subasta"))
        frames.append(df)
    merged_exports = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["Nombre Subasta","Fecha Subasta","Ver vínculo del lote"])
    try:
        df_fallback = pd.read_excel(fallback_path)
        url_col = None
        for c in df_fallback.columns:
            if str(c).strip().lower() in {"link_ficha","link","url","enlace"}:
                url_col = c; break
        if url_col is None and df_fallback.shape[1] == 1:
            url_col = df_fallback.columns[0]
        if url_col is None:
            df_fallback_clean = pd.DataFrame(columns=["Ver vínculo del lote"])
        else:
            df_fallback_clean = pd.DataFrame({"Ver vínculo del lote": df_fallback[url_col].astype(str).str.strip()})
        df_fallback_clean.insert(0, "Nombre Subasta", None)
        df_fallback_clean.insert(1, "Fecha Subasta", None)
    except Exception:
        df_fallback_clean = pd.DataFrame(columns=["Nombre Subasta","Fecha Subasta","Ver vínculo del lote"])
    union = pd.concat([merged_exports, df_fallback_clean], ignore_index=True, sort=False)
    if "Ver vínculo del lote" not in union.columns: union["Ver vínculo del lote"] = None
    canon = union["Ver vínculo del lote"].astype(str).map(canonicalize_lot_url)
    union = union[~canon.duplicated(keep="first")].copy()
    cols = list(union.columns)
    for key in ["Nombre Subasta", "Fecha Subasta"]:
        if key in cols: cols.remove(key)
    cols = ["Nombre Subasta", "Fecha Subasta"] + cols
    union = union[cols]
    union.to_excel(out_path, index=False)
    return out_path, len(union)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Argumentos opcionales
    exports_source = sys.argv[1] if len(sys.argv) > 1 else None
    fallback_path  = sys.argv[2] if len(sys.argv) > 2 else None
    out_path       = sys.argv[3] if len(sys.argv) > 3 else f"fichas_vehiculos_{datetime.now():%Y%m%d}.xlsx"

    # Autodetección de origen de exports si no se pasa arg1
    if not exports_source:
        if Path("_output_excels").exists():
            exports_source = "_output_excels"
        elif Path("_output_excels.zip").exists():
            exports_source = "_output_excels.zip"
        else:
            raise FileNotFoundError("No encuentro ni la carpeta '_output_excels' ni el zip '_output_excels.zip'.")

    # Autodetección del fallback si no se pasa arg2
    if not fallback_path:
        # Busca por patrón común (hoy) o el último fichas_vehiculos_*.xlsx
        candidates = sorted(Path(".").glob("fichas_vehiculos_*.xlsx"), reverse=True)
        if candidates:
            fallback_path = str(candidates[0])
        else:
            raise FileNotFoundError("No encuentro el Excel de fallback (p.ej. 'fichas_vehiculos_YYYYMMDD.xlsx').")

    run_merge(exports_source, fallback_path, out_path)