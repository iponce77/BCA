
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

    def _pick_model_base_col(self) -> str:
        for c in ["modelo_base_x","modelo_base","modelo_base_y","modelo_base_match","modelo"]:
            if c in self.df.columns:
                return c
        return "modelo"

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
        grp_keys = ["marca", self._pick_model_base_col(), "anio","combustible_norm","sale_country"]
        self.price_mean_country = (
            self.df.groupby(grp_keys, dropna=False)["precio_final_eur"]
            .mean()
            .rename("precio_medio_modelo_pais")
        )
        grp_keys2 = grp_keys + ["sale_name"]
        self.price_mean_auction = (
            self.df.groupby(grp_keys2, dropna=False)["precio_final_eur"]
            .mean()
            .rename("precio_medio_modelo_pais_subasta")
        )

    # ------------------------- Utilidades internas -------------------------
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
            return pd.Series(1.0, index=self.df.index)
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        stack = np.vstack([p.fillna(0.0).to_numpy() for p in parts])
        return pd.Series((w @ stack), index=self.df.index)

    def _fast_rotation_proxy(self, region: str) -> pd.Series:
        mix_col = self._region_field("mix_0_3_%", region)
        rank_col = self._region_field("rank_year_model", region)
        if mix_col in self.df.columns:
            s = self.df[mix_col].astype(float)
            if s.max() > 1.0:
                s = s / 100.0
            return self._normalize(s)
        elif rank_col in self.df.columns:
            s = self.df[rank_col].astype(float)
            # rank bajo = mejor → invertimos
            return 1.0 - self._normalize(s)
        else:
            return pd.Series(1.0, index=self.df.index)

    def _composite_score(self, region: str) -> pd.Series:
        # margen normalizado
        mar = self._normalize(self.df["margin_abs"].astype(float))
        # demanda
        dem = self._demand_factor(region)
        # rotación
        rot = self._fast_rotation_proxy(region)
        # pesos
        alpha = float(self.cfg.alpha_margin)
        rot_weight = (1.0 - alpha) + float(self.cfg.rotation_boost)
        # ensamblado: margen y (demanda*rot)
        return (alpha * mar) + (rot_weight * dem * rot)

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
            df = df[pd.to_numeric(df["precio_final_eur"], errors="coerce") <= float(max_price)]
        if min_price is not None:
            df = df[pd.to_numeric(df["precio_final_eur"], errors="coerce") >= float(min_price)]

        # filtros por año exacto (lista o escalar)
        if year_exact is not None and "anio" in df.columns:
            if isinstance(year_exact, (list, tuple, set)):
                yset = {int(y) for y in year_exact}
                df = df[df["anio"].isin(yset)]
            else:
                df = df[df["anio"] == int(year_exact)]
        # Segmento include/exclude (si existe columna segmento/segment)
        seg_col = next((c for c in ["segmento","segment","segmento_norm"] if c in df.columns), None)
        if seg_col:
            if segment_include is not None:
                if not isinstance(segment_include,(list,tuple,set)):
                    segment_include = [segment_include]
                vals = {str(x).strip().upper() for x in segment_include}
                df = df[df[seg_col].astype(str).str.upper().isin(vals)]
            if segment_exclude is not None:
                if not isinstance(segment_exclude,(list,tuple,set)):
                    segment_exclude = [segment_exclude]
                vals = {str(x).strip().upper() for x in segment_exclude}
                df = df[~df[seg_col].astype(str).str.upper().isin(vals)]
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

        df = df.copy()
        df["score"] = score.loc[df.index].fillna(0.0)

        # si solo queremos agrupar por marca (exploración), lo soportamos
        if brand_only:
            agg = df.groupby(["marca"], dropna=False).agg(
                precio_medio=("precio_final_eur","mean"),
                margin_abs_medio=("margin_abs","mean"),
                n_listings=("precio_final_eur","size"),
                score=("score","mean"),
            ).reset_index()
            agg = agg.sort_values(["score","margin_abs_medio"] if not ignore_rotation else ["margin_abs_medio","score"], ascending=False)
            return agg.head(n)

        # claves de agrupación estándar (marca, modelo_base, año, combustible)
        group_keys = ["marca", self._pick_model_base_col(), "anio", "combustible_norm"]

        # descartar grupos con pocas muestras (si se pidió)
        if min_listings_per_group > 1:
            sizes = df.groupby(group_keys, dropna=False)[self._pick_model_base_col()].size().rename("n_listings").reset_index()
            df = df.merge(sizes, on=group_keys, how="left")
            df = df[df["n_listings"] >= int(min_listings_per_group)].drop(columns=["n_listings"])

        if selection == "cheapest":
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
                   .quantile([0.25,0.75]).unstack().rename(columns={0.25:"p25",0.75:"p75"}))
            out = out.merge(iqr.reset_index(), on=group_keys, how="left")
            out = out.assign(
                score=out["score"].fillna(0.0)
            )
            if prefer_cheapest_sort:
                sort_cols = ["precio_min","score"]
                out = out.sort_values(sort_cols, ascending=[True, False])
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
                rank_model_region=(self._region_field("rank_year_model", region),"mean") if (self._region_field("rank_year_model", region)
 in df.columns) else ("score","mean"),
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
            if not include_sale_country:
                agg = agg.drop(columns=["sale_country"]) if "sale_country" in agg.columns else agg
            if not include_sale_name:
                agg = agg.drop(columns=["sale_name"]) if "sale_name" in agg.columns else agg
            return agg.head(n)

    def query_1(self) -> pd.DataFrame:
        # placeholder para compatibilidad; Q1 real se gestiona desde run_queries con filtros específicos
        return self.recommend_best(region="esp", ignore_rotation=True, n=10)

    def query_2(self) -> pd.DataFrame:
        return self.recommend_best(region="bcn", selection="cheapest", prefer_cheapest_sort=True, n=15)

    def query_3(self) -> pd.DataFrame:
        return self.recommend_best(region="bcn", ignore_rotation=True, n=10)

    def query_4(self) -> pd.DataFrame:
        return self.recommend_best(region="cat", min_price=20000, max_price=25000, prefer_fast=True, n=10)

    def query_5(self) -> Dict[str, pd.DataFrame]:
        sub = self.df[(self.df["marca"].str.upper()=="SEAT") & (self.df[self._pick_model_base_col()].str.upper()=="LEON") & (self.df["anio"]==2023)]
        if sub.empty:
            return {"by_country": pd.DataFrame(), "by_subasta": pd.DataFrame()}

        k_country = ["marca", self._pick_model_base_col(), "anio","combustible_norm","sale_country"]
        by_country = (
            sub.groupby(k_country, dropna=False)["precio_final_eur"]
            .mean().rename("precio_medio_modelo_pais")
            .reset_index()
            .sort_values("precio_medio_modelo_pais")
        )
        k_sub = ["marca", self._pick_model_base_col(), "anio","combustible_norm","sale_country","sale_name"]
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
            df[self._pick_model_base_col()].astype(str).str.contains(str(model), case=False, na=False)
            & (df["anio"] == int(year))
        )
        sub = df[mask]
        if sub.empty:
            return {"by_country": pd.DataFrame(), "by_subasta": pd.DataFrame()}

        k_country = ["marca", self._pick_model_base_col(), "anio","combustible_norm","sale_country"]
        by_country = (
            sub.groupby(k_country, dropna=False)["precio_final_eur"]
            .mean().rename("precio_medio_modelo_pais")
            .reset_index()
            .sort_values("precio_medio_modelo_pais")
        )
        k_sub = ["marca", self._pick_model_base_col(), "anio","combustible_norm","sale_country","sale_name"]
        by_subasta = (
            sub.groupby(k_sub, dropna=False)["precio_final_eur"]
            .mean().rename("precio_medio_modelo_pais_subasta")
            .reset_index()
            .sort_values(["precio_medio_modelo_pais_subasta","sale_country","sale_name"])
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
