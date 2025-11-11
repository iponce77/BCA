
from __future__ import annotations
from dataclasses import dataclass, field
from dataclasses import dataclass
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
    weight_concentration: float = 0.0

@dataclass
class RecommenderConfig:
    # weight knobs for composite scoring (0..1)
    alpha_margin: float = 0.6   # alpha (margen) ; rotación = 1 - alpha
    rotation_boost: float = 0.0
    demand: DemandConfig = field(default_factory=DemandConfig)

class BCAInvestRecommender:
    def __init__(self, df: pd.DataFrame, cfg: Optional[RecommenderConfig] = None):
        self.df = df.copy()
        self.cfg = cfg or RecommenderConfig()
        self._normalize_types()
        self._check_columns_minimum()
        self._build_price_means()

    # ------------------------- Core prep -------------------------
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
        # normalizar posibles columnas de kms y URL si existen
        for c in ["km","kilometros","mileage","odometro"]:
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
        grp_keys = ["marca","modelo","anio","combustible_norm","sale_country"]
        self.price_mean_country = (
            self.df.groupby(grp_keys, dropna=False)["precio_final_eur"]
            .mean()
            .rename("precio_medio_modelo_pais")
            .reset_index()
        )
        grp_keys_sub = ["marca","modelo","anio","combustible_norm","sale_country","sale_name"]
        self.price_mean_subasta = (
            self.df.groupby(grp_keys_sub, dropna=False)["precio_final_eur"]
            .mean()
            .rename("precio_medio_modelo_pais_subasta")
            .reset_index()
        )

    # ------------------------- Helpers -------------------------
    def _region_field(self, base: str, region: str) -> str:
        return f"{base}_{region}"

    def _fast_rotation_proxy(self, region: str) -> pd.Series:
        col_mix = self._region_field("mix_0_3_%", region)
        if col_mix in self.df.columns:
            s = self.df[col_mix].astype(float)
            if s.max() > 1.0:
                s = s / 100.0
            return s.fillna(0.0)
        col_rank = self._region_field("rank_year_model", region)
        if col_rank in self.df.columns:
            r = self.df[col_rank].astype(float)
            return (1.0 / (1.0 + r)).fillna(0.0)
        return pd.Series(0.0, index=self.df.index)

    def _normalize(self, s: pd.Series) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            return s.fillna(0.0)
        a = s.min()
        b = s.max()
        if pd.isna(a) or pd.isna(b) or a==b:
            return s.fillna(0.0)
        return (s - a) / (b - a)

    def _demand_factor(self, region: str) -> pd.Series:
        dc = self.cfg.demand
        parts = []
        weights = []
        if dc.use_brand_share:
            col = self._region_field("share_marca_%", region)
            if col in self.df.columns:
                s = self.df[col].astype(float)
                if s.max() > 1.0:
                    s = s / 100.0
                parts.append(self._normalize(s))
                weights.append(dc.weight_brand_share)
        if dc.use_units_abs:
            col = self._region_field("units_abs", region)
            if col in self.df.columns:
                parts.append(self._normalize(self.df[col].astype(float)))
                weights.append(dc.weight_units_abs)
        if dc.use_concentration_penalty:
            col = self._region_field("dominancia_modelo_marca_%", region)
            if col in self.df.columns:
                s = self.df[col].astype(float)
                if s.max() > 1.0:
                    s = s / 100.0
                parts.append(1.0 - self._normalize(s))
                weights.append(dc.weight_concentration)

        if not parts:
            return pd.Series(1.0, index=self.df.index)  # neutral
        wsum = sum(weights) if sum(weights) != 0 else 1.0
        demand = sum(p*w for p,w in zip(parts, weights)) / wsum
        return demand.clip(0,1).fillna(0.0)

    def _composite_score(self, region: str) -> pd.Series:
        alpha = float(self.cfg.alpha_margin)
        rot_weight = max(0.0, min(1.0, 1.0 - alpha + self.cfg.rotation_boost))
        mar_weight = max(0.0, min(1.0, alpha))

        margin = self._normalize(self.df["margin_abs"])
        demand = self._demand_factor(region)
        margin_demand = margin * demand  # "foto" del mercado

        rot = self._normalize(self._fast_rotation_proxy(region))

        score = mar_weight * margin_demand + rot_weight * rot
        return score

    # ------------------------- Public API -------------------------
    def recommend_best(self,
                       region: str,
                       max_age_years: Optional[int] = None,
                       max_price: Optional[float] = None,
                       min_price: Optional[float] = None,
                       ignore_rotation: bool = False,
                       prefer_fast: bool = False,
                       brand_only: bool = False,
                       # NUEVO: selección y filtros avanzados
                       selection: str = "mean",  # "mean" | "cheapest"
                       year_exact: Optional[Union[int, List[int]]] = None,
                       segment_include: Optional[Union[str, List[str]]] = None,
                       segment_exclude: Optional[Union[str, List[str]]] = None,
                       mileage_min: Optional[float] = None,
                       mileage_max: Optional[float] = None,
                       include_sale_country: bool = True,
                       include_sale_name: bool = True,
                       min_listings_per_group: int = 1,
                       prefer_cheapest_sort: bool = False,
                       n: int = 10) -> pd.DataFrame:
        if region not in REGIONS:
            raise ValueError(f"region no válida: {region}. Usa {REGIONS}")
        df = self.df.copy()

        # filtros de edad y precio
        if max_age_years is not None and "anio" in df.columns:
            from datetime import datetime
            current_year = datetime.now().year
            min_year = current_year - max_age_years
            df = df[df["anio"] >= min_year]

        if max_price is not None:
            df = df[df["precio_final_eur"] <= max_price]
        if min_price is not None:
            df = df[df["precio_final_eur"] >= min_price]

        # --- NUEVOS FILTROS ---
        # Año exacto (int o lista)
        if year_exact is not None and "anio" in df.columns:
            if isinstance(year_exact, (list, tuple, set)):
                df = df[df["anio"].isin([int(x) for x in year_exact])]
            else:
                df = df[df["anio"] == int(year_exact)]
        # Segmento include/exclude (si existe columna segmento/segment)
        seg_col = next((c for c in ["segmento","segment","segmento_norm"] if c in df.columns), None)
        if seg_col:
            if segment_include is not None:
                vals = {str(x).strip().upper() for x in (segment_include if isinstance(segment_include,(list,tuple,set)) else [segment_include])}
                df = df[df[seg_col].astype(str)
                                   .str.normalize("NFKD")
                                   .str.encode("ascii", "ignore")
                                   .str.decode("ascii")
                                   .str.upper()
                                   .isin(vals)]  
            if segment_exclude is not None:
                vals = {str(x).strip().upper() for x in (segment_exclude if isinstance(segment_exclude,(list,tuple,set)) else [segment_exclude])}
                df = df[~df[seg_col].astype(str).str.normalize("NFKD").str.encode("ascii","ignore").str.decode().str.upper().isin(vals)]
        # Kilometraje (si existe columna)
        km_col = next((c for c in ["km","kilometros","kilómetros","mileage","odometro","odómetro"] if c in df.columns), None)
        if km_col:
            if mileage_min is not None:
                df = df[pd.to_numeric(df[km_col], errors="coerce") >= float(mileage_min)]
            if mileage_max is not None:
                df = df[pd.to_numeric(df[km_col], errors="coerce") <= float(mileage_max)]

        # ajustar pesos por flags
        prev_cfg = self.cfg
        cfg = RecommenderConfig(
            alpha_margin=1.0 if ignore_rotation else prev_cfg.alpha_margin,
            rotation_boost=(0.3 if prefer_fast else 0.0),
            demand=prev_cfg.demand
        )
        self.cfg = cfg
        score = self._composite_score(region)
        self.cfg = prev_cfg  # restore

        df = df.assign(score=score)

        group_keys = ["marca","modelo","anio","combustible_norm"]
        if include_sale_country: group_keys.append("sale_country")
        if include_sale_name:    group_keys.append("sale_name")

        # descartar grupos con pocas muestras (si se pidió)
        if min_listings_per_group > 1:
            sizes = df.groupby(group_keys, dropna=False)["modelo"].size().rename("n_listings").reset_index()
            df = df.merge(sizes, on=group_keys, how="left")
            df = df[df["n_listings"] >= int(min_listings_per_group)].drop(columns=["n_listings"])

        if selection.lower() == "cheapest":
            # ordenar por precio y tomar la fila mínima por grupo (tamaño mínimo ya aplicado arriba)
            # Sanea precio y controla vacío tras filtros
            df["precio_final_eur"] = pd.to_numeric(df["precio_final_eur"], errors="coerce")
            df = df[df["precio_final_eur"].notna()]
            if df.empty:
                return df.head(0)

            # Ordénalo por precio asc y, de empate, por score desc (mejor candidato)
            df_sorted = df.sort_values(["precio_final_eur", "score"], ascending=[True, False])
            picked = df_sorted.groupby(group_keys, dropna=False, as_index=False).first()

            out = picked.rename(columns={
                "precio_final_eur": "precio_min",
                "margin_abs": "margin_abs_min",
            })
            # km, link, lot_id si existen
            if km_col and km_col in out.columns:
                out = out.rename(columns={km_col: "km_min"})
            link_col = next((c for c in ["url","link","lote_url","listing_url","link_ficha"] if c in out.columns), None)
            if link_col:
                out = out.rename(columns={link_col: "link_ficha_min"})
            lot_col = next((c for c in ["lot_id","lote_id","listing_id","id_bca"] if c in out.columns), None)
            if lot_col:
                out = out.rename(columns={lot_col: "lot_id_min"})
            # segmento si existe
            if seg_col and seg_col in out.columns:
                out = out.rename(columns={seg_col: "segmento"})
            # rotación/demanda informativas
            mix_col  = self._region_field("mix_0_3_%", region)
            rank_col = self._region_field("rank_year_model", region)
            if mix_col in out.columns:
                out["mix_0_3"] = (out[mix_col] / 100.0) if out[mix_col].max() > 1.0 else out[mix_col]
            if rank_col in out.columns:
                out["rank_model_region"] = out[rank_col]
            # IQR de precio por cluster
            iqr = (df.groupby(group_keys, dropna=False)["precio_final_eur"]
                     .agg(lambda s: (s.quantile(0.75) - s.quantile(0.25)))
                     .rename("precio_iqr")).reset_index()
            out = out.merge(iqr, on=group_keys, how="left")
            # ordenar
            if prefer_cheapest_sort:
                out = out.sort_values(["precio_min"], ascending=True)
            else:
                sort_cols = ["score","margin_abs_min"] if not ignore_rotation else ["margin_abs_min","score"]
                out = out.sort_values(sort_cols, ascending=False)
            return out.head(n)
        else:
            # camino clásico: medias por cluster
            agg = df.groupby(group_keys, dropna=False).agg(
                precio_medio=("precio_final_eur","mean"),
                margin_abs_medio=("margin_abs","mean"),
                # Preferir margin_pct si existe; si no, margin_ptc; si no, placeholder
                margin_ptc_medio=(("margin_pct","mean") if "margin_pct" in df.columns
                                  else (("margin_ptc","mean") if "margin_ptc" in df.columns
                                        else ("margin_abs","mean"))),                n_listings=("precio_final_eur","size"),
                score=("score","mean"),
                mix_0_3=(self._region_field("mix_0_3_%", region),"mean") if (self._region_field("mix_0_3_%", region) in df.columns) else ("score","mean"),
                rank_model_region=(self._region_field("rank_year_model", region),"mean") if (self._region_field("rank_year_model", region) in df.columns) else ("score","mean"),
            ).reset_index()
            # Si no existe margin_pct/margin_ptc en el dataset, calcular % medio como mean(margin_abs / precio_final_eur) por cluster
            if ("margin_pct" not in df.columns and "margin_ptc" not in df.columns
                and "precio_medio" in agg.columns and "margin_ptc_medio" in agg.columns):
                _gkey = "__g__"
                _df = df.copy()
                _df[_gkey] = _df[group_keys].astype(str).agg("|".join, axis=1)
                frac = (_df["margin_abs"] / _df["precio_final_eur"]).replace([np.inf, -np.inf], np.nan)
                mean_frac = _df.assign(__frac=frac).groupby(_gkey)["__frac"].mean()
                agg[_gkey] = agg[group_keys].astype(str).agg("|".join, axis=1)
                agg["margin_ptc_medio"] = agg[_gkey].map(mean_frac)
                agg.drop(columns=[_gkey], inplace=True) 

            sort_cols = ["score","margin_abs_medio"] if not ignore_rotation else ["margin_abs_medio","score"]
            agg = agg.sort_values(sort_cols, ascending=False)

        if brand_only:
            b = df.groupby("marca", dropna=False).agg(
                mean_score=("score","mean"),
                total_margin=("margin_abs","sum"),
                listings=("precio_final_eur","size")
            ).reset_index().sort_values(["mean_score","total_margin"], ascending=False)
            return b.head(n)

        return agg.head(n)

    def query_1(self) -> pd.DataFrame:
        return self.recommend_best(region="bcn", max_age_years=5, max_price=15000, prefer_fast=True, n=10)

    def query_2(self) -> pd.DataFrame:
        return self.recommend_best(region="bcn", brand_only=True, n=10)

    def query_3(self) -> pd.DataFrame:
        return self.recommend_best(region="bcn", ignore_rotation=True, n=10)

    def query_4(self) -> pd.DataFrame:
        return self.recommend_best(region="cat", min_price=20000, max_price=25000, prefer_fast=True, n=10)

    def query_5(self) -> Dict[str, pd.DataFrame]:
        sub = self.df[(self.df["marca"].str.upper()=="SEAT") & (self.df["modelo"].str.upper()=="LEON") & (self.df["anio"]==2023)]
        if sub.empty:
            return {"by_country": pd.DataFrame(), "by_subasta": pd.DataFrame()}

        k_country = ["marca","modelo","anio","combustible_norm","sale_country"]
        by_country = (
            sub.groupby(k_country, dropna=False)["precio_final_eur"]
            .mean().rename("precio_medio_modelo_pais")
            .reset_index()
            .sort_values("precio_medio_modelo_pais")
        )
        k_sub = ["marca","modelo","anio","combustible_norm","sale_country","sale_name"]
        by_subasta = (
            sub.groupby(k_sub, dropna=False)["precio_final_eur"]
            .mean().rename("precio_medio_modelo_pais_subasta")
            .reset_index()
            .sort_values(["precio_medio_modelo_pais_subasta","sale_country","sale_name"])
        )
        return {"by_country": by_country, "by_subasta": by_subasta}

    def query_special(self, model: str, year: int):
        df = self.df.copy()
        mask = (
            df["modelo"].astype(str).str.contains(str(model), case=False, na=False)
            & (df["anio"] == int(year))
        )
        sub = df[mask]
        if sub.empty:
            return {"by_country": pd.DataFrame(), "by_subasta": pd.DataFrame()}

        by_country = (
            sub.groupby(["sale_country","marca","modelo","anio","combustible_norm"], dropna=False)
               .agg(precio_medio=("precio_final_eur","mean"),
                    margin_abs_medio=("margin_abs","mean"),
                    n_listings=("modelo","size"))
               .reset_index()
               .sort_values(["precio_medio","sale_country","modelo","anio"])
        )

        by_subasta = (
            sub.groupby(["sale_country","sale_name","marca","modelo","anio","combustible_norm"], dropna=False)
               .agg(precio_medio=("precio_final_eur","mean"),
                    margin_abs_medio=("margin_abs","mean"),
                    n_listings=("modelo","size"))
               .reset_index()
               .sort_values(["precio_medio","sale_country","sale_name","modelo","anio"])
        )

        return {"by_country": by_country, "by_subasta": by_subasta}

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
