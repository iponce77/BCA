from __future__ import annotations
import os, re, logging, json, polars as pl, pandas as pd, yaml

def _strip_upper_ascii(s: str) -> str:
    import unicodedata
    if s is None: return ""
    s2 = str(s).upper().strip()
    s2 = ''.join(ch for ch in unicodedata.normalize('NFD', s2) if unicodedata.category(ch) != 'Mn')
    s2 = re.sub(r"[^0-9A-Z \-]", " ",s2)
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def _pairs_from_yaml(doc: dict):
    brand_pairs=[]; fuel_pairs=[]
    for item in (doc.get("brands") or []):
        canon=_strip_upper_ascii(item.get("canonical",""))
        for a in (item.get("aliases") or []):
            brand_pairs.append({"alias_norm": _strip_upper_ascii(a), "marca_normalizada": canon})
    for item in (doc.get("fuels") or []):
        canon=_strip_upper_ascii(item.get("canonical",""))
        for a in (item.get("aliases") or []):
            fuel_pairs.append({"alias_norm": _strip_upper_ascii(a), "combustible": canon})
    bdf = pl.DataFrame(brand_pairs) if brand_pairs else pl.DataFrame({"alias_norm":[], "marca_normalizada":[]})
    fdf = pl.DataFrame(fuel_pairs) if fuel_pairs else pl.DataFrame({"alias_norm":[], "combustible":[]})
    return bdf, fdf

def load_yaml(path: str | None):
    if not path or not os.path.exists(path):
        return pl.DataFrame({"alias_norm":[], "marca_normalizada":[]}), pl.DataFrame({"alias_norm":[], "combustible":[]})
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        return _pairs_from_yaml(doc)
    except Exception as e:
        logging.warning(f"[mappings] YAML inválido: {path} ({e})")
        return pl.DataFrame({"alias_norm":[], "marca_normalizada":[]}), pl.DataFrame({"alias_norm":[], "combustible":[]})

def load_fuel_json(path: str | None) -> pl.DataFrame:
    if not path or not os.path.exists(path):
        return pl.DataFrame({"alias_norm":[], "combustible":[]})
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f) or {}
        to5={"diesel":"DIESEL","petrol":"GASOLINA","hev":"HEV","phev":"PHEV","electric":"BEV"}
        rows=[]
        for alias, meta in doc.items():
            canon=(meta or {}).get("canonical_fuel","").lower().strip()
            rows.append({"alias_norm": _strip_upper_ascii(alias), "combustible": to5.get(canon, "OTROS")})
            rows.append({"alias_norm": _strip_upper_ascii(canon), "combustible": to5.get(canon, "OTROS")})
        return pl.DataFrame(rows) if rows else pl.DataFrame({"alias_norm":[], "combustible":[]})
    except Exception as e:
        logging.warning(f"[mappings] JSON combustible inválido: {path} ({e})")
        return pl.DataFrame({"alias_norm":[], "combustible":[]})

def load_whitelist_brands(path: str | None) -> pl.DataFrame:
    if not path or not os.path.exists(path):
        return pl.DataFrame({"alias_norm":[], "marca_normalizada":[]})
    try:
        df = pd.read_excel(path)
        marca_col = next((c for c in df.columns if str(c).lower().startswith("marca")), df.columns[0])
        brands = sorted({ _strip_upper_ascii(x) for x in df[marca_col].dropna().astype(str).tolist() })
        return pl.DataFrame({"alias_norm": brands, "marca_normalizada": brands})
    except Exception as e:
        logging.warning(f"[mappings] No pude leer whitelist: {path} ({e})")
        return pl.DataFrame({"alias_norm":[], "marca_normalizada":[]})

def merge_mappings(bdf_yaml: pl.DataFrame, bdf_wl: pl.DataFrame, fdf_yaml: pl.DataFrame, fdf_json: pl.DataFrame):
    b_all = pl.concat([bdf_yaml, bdf_wl], how="vertical_relaxed").unique() if (bdf_yaml.height + bdf_wl.height)>0 else pl.DataFrame({"alias_norm":[], "marca_normalizada":[]})
    f_all = pl.concat([fdf_json, fdf_yaml], how="vertical_relaxed").unique() if (fdf_yaml.height + fdf_json.height)>0 else pl.DataFrame({"alias_norm":[], "combustible":[]})
    return b_all, f_all

def load_all(mappings_yaml: str | None, fuel_json: str | None, whitelist_xlsx: str | None):
    byaml, fyaml = load_yaml(mappings_yaml)
    fjson = load_fuel_json(fuel_json)
    bwl = load_whitelist_brands(whitelist_xlsx)
    bdf, fdf = merge_mappings(byaml, bwl, fyaml, fjson)
    if bdf.height == 0:
        bdf = pl.DataFrame([
            {"alias_norm":"MERCEDES-BENZ","marca_normalizada":"MERCEDES BENZ"},
            {"alias_norm":"MERCEDES BENZ","marca_normalizada":"MERCEDES BENZ"},
            {"alias_norm":"MERCEDES","marca_normalizada":"MERCEDES BENZ"},
            {"alias_norm":"MB","marca_normalizada":"MERCEDES BENZ"},
            {"alias_norm":"VW","marca_normalizada":"VOLKSWAGEN"},
            {"alias_norm":"VOLKSWAGEN","marca_normalizada":"VOLKSWAGEN"},
            {"alias_norm":"ŠKODA","marca_normalizada":"SKODA"},
            {"alias_norm":"SKODA","marca_normalizada":"SKODA"},
        ]).with_columns(pl.col("alias_norm").str.to_uppercase())
    if fdf.height == 0:
        fdf = pl.DataFrame([
            {"alias_norm":"DIESEL","combustible":"DIESEL"},
            {"alias_norm":"GASOIL","combustible":"DIESEL"},
            {"alias_norm":"GASOLINA","combustible":"GASOLINA"},
            {"alias_norm":"PETROL","combustible":"GASOLINA"},
            {"alias_norm":"HEV","combustible":"HEV"},
            {"alias_norm":"MHEV","combustible":"HEV"},
            {"alias_norm":"PHEV","combustible":"PHEV"},
            {"alias_norm":"PLUG IN","combustible":"PHEV"},
            {"alias_norm":"BEV","combustible":"BEV"},
            {"alias_norm":"EV","combustible":"BEV"},
        ])
    return {"brands_df": bdf, "fuels_df": fdf, "brands_n": bdf.height, "fuels_n": fdf.height}
