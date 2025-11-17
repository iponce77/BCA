from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

REGIONS = ["bcn","cat","esp"]

@dataclass
class DemandConfig:
    use_brand_share: bool = True
    use_units_abs: bool = True
    use_concentration_penalty: bool = False
    weight_brand_share: float = 0.5
    weight_units_abs: float = 0.5
    weight_concentration: float = 0.25

@dataclass
class RecommenderConfig:
    alpha_margin: float = 0.6
    rotation_boost: float = 0.0
    demand: DemandConfig = field(default_factory=DemandConfig)

class BCAInvestRecommender:
    def __init__(self, df: pd.DataFrame, cfg: Optional[RecommenderConfig] = None):
        self.df = df.copy()
        self.cfg = cfg or RecommenderConfig()
        self._normalize_types()
        self._check_columns_minimum()
        self._build_price_means()

    # ------------------------- helpers -------------------------
    def _pick_model_base_col(self) -> str:
        for c in ["modelo_base_x","modelo_base","modelo_base_y","modelo_base_match","modelo"]:
            if c in self.df.columns:
                return c
        return "modelo"

    def _normalize_types(self):
        for c in ["anio"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        if "combustible_norm" in self.df.columns:
            self.df["combustible_norm"] = self.df["combustible_norm"].astype(str).str.upper().str.strip()
        for c in ["precio_final_eur", "precio_venta_ganvam", "margin_abs", "margin_ptc", "margin_pct"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        if "sale_name" in self.df.columns:
            self.df["sale_name"] = self.df["sale_name"].astype(str).str.strip()
        for c in ["km","kilometros","mileage","odometro","odómetro"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        for c in ["url","link","lote_url","listing_url","link_ficha"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).str.strip()
        for c in ["lot_id","lote_id","listing_id","id_bca"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).str.strip()

    def _check_columns_minimum(self):
        required = [
            "marca","modelo","anio",
            "combustible_norm",
            "precio_final_eur","precio_venta_ganvam",
            "margin_abs",
            "sale_country","sale_name",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

    def _build_price_means(self):
        grp_keys = ["marca", self._pick_model_base_col(), "anio","combustible_norm","sale_country"]
        self.price_mean_country = (
            self.df.groupby(grp_keys, dropna=False)["precio_final_eur"].mean().rename("precio_medio_modelo_pais")
        )
        grp_keys2 = grp_keys + ["sale_name"]
        self.price_mean_auction = (
            self.df.groupby(grp_keys2, dropna=False)["precio_final_eur"].mean().rename("precio_medio_modelo_pais_subasta")
        )

    def _region_field(self, base: str, region: str) -> str:
        if base.endswith("_%"):
            return f"{base}_{region}"
        elif base.endswith("_"):
            return f"{base}{region}"
        else:
            return f"{base}_{region}"

    def _normalize(self, s: pd.Series) -> pd.Series:
        s = s.astype(float)
        a = s.min()
        b = s.max()
        if pd.isna(a) or pd.isna(b) or a==b:
            return s.fillna(0.0)
        return (s - a) / (b - a)

    def _demand_factor(self, region: str) -> pd.Series:
        dc = self.cfg.demand
        parts = []; weights = []
        if dc.use_brand_share:
            col = self._region_field("share_marca_%", region)
            if col in self.df.columns:
                s = self.df[col].astype(float)
                if s.max() > 1.0: s = s / 100.0
                parts.append(self._normalize(s)); weights.append(dc.weight_brand_share)
        if dc.use_units_abs:
            col = self._region_field("units_abs", region)
            if col in self.df.columns:
                parts.append(self._normalize(self.df[col].astype(float))); weights.append(dc.weight_units_abs)
        if dc.use_concentration_penalty:
            col = self._region_field("dominancia_modelo_marca_%", region)
            if col in self.df.columns:
                s = self.df[col].astype(float)
                if s.max() > 1.0: s = s / 100.0
                parts.append(1.0 - self._normalize(s)); weights.append(dc.weight_concentration)
        if not parts:
            return pd.Series(1.0, index=self.df.index)
        w = np.array(weights, dtype=float); w = w / w.sum()
        stack = np.vstack([p.fillna(0.0).to_numpy() for p in parts])
        return pd.Series((w @ stack), index=self.df.index)

    def _fast_rotation_proxy(self, region: str) -> pd.Series:
        mix_col = self._region_field("mix_0_3_%", region)
        rank_col = self._region_field("rank_year_model", region)
        if mix_col in self.df.columns:
            s = self.df[mix_col].astype(float)
            if s.max() > 1.0: s = s / 100.0
            return self._normalize(s)
        elif rank_col in self.df.columns:
            s = self.df[rank_col].astype(float)
            return 1.0 - self._normalize(s)
        else:
            return pd.Series(1.0, index=self.df.index)

    def _composite_score(self, region: str) -> pd.Series:
        mar = self._normalize(self.df["margin_abs"].astype(float))
        dem = self._demand_factor(region)
        rot = self._fast_rotation_proxy(region)
        alpha = float(self.cfg.alpha_margin)
        rot_weight = (1.0 - alpha) + float(self.cfg.rotation_boost)
        return (alpha * mar) + (rot_weight * dem * rot)

    # ------------------------- Q3 específica -------------------------
    def q3_segment_price_order(self,
                               region: str,
                               segment: str,
                               n: int = 20,
                               min_year: int = 2020,
                               max_year: int = 2025,
                               mileage_max: Optional[float] = 100000,
                               fuel_include: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """Dentro de un segmento: pick del más barato por cluster y ordenar por margen desc, Top-N."""
        if region not in REGIONS:
            raise ValueError(f"region no válida: {region}. Usa {REGIONS}")
        df = self.df.copy()

        seg_col = next((c for c in ["segmento","segment","segmento_norm"] if c in df.columns), None)
        if not seg_col:
            raise ValueError("No existe columna segmento/segment/segmento_norm en el dataset")

        # Filtros base
        df = df[df[seg_col].astype(str).str.upper() == str(segment).strip().upper()]
        if "anio" in df.columns:
            df = df[(df["anio"] >= int(min_year)) & (df["anio"] <= int(max_year))]
        km_col = next((c for c in ["km","kilometros","kilómetros","mileage","odometro","odómetro"] if c in df.columns), None)
        if mileage_max is not None and km_col:
            df = df[pd.to_numeric(df[km_col], errors="coerce") <= float(mileage_max)]
        if fuel_include is not None and "combustible_norm" in df.columns:
            vals = {str(x).strip().upper() for x in (fuel_include if isinstance(fuel_include,(list,tuple,set)) else [fuel_include])}
            df = df[df["combustible_norm"].astype(str).str.upper().isin(vals)]

        # score informativo
        score = self._composite_score(region)
        df = df.copy(); df["score"] = score.loc[df.index].fillna(0.0)

        # pick más barato por cluster (marca, modelo_base, año, combustible)
        group_keys = ["marca", self._pick_model_base_col(), "anio", "combustible_norm"]
        df_sorted = df.sort_values(["precio_final_eur", "margin_abs", "score"], ascending=[True, False, False])
        cheapest_per_cluster = df_sorted.groupby(group_keys, dropna=False, as_index=False).first()

        # Top-N por margen (desc) sin duplicar cluster (ya es único por group_keys)
        topn = cheapest_per_cluster.sort_values(["margin_abs","score"], ascending=[False, False]).head(int(n))
        return topn

# ------------------------- loader -------------------------
def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx",".xls"]:
        return pd.read_excel(path)
    elif path.suffix.lower() in [".csv",".txt"]:
        return pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError("Formato no soportado. Usa .parquet, .xlsx o .csv")

