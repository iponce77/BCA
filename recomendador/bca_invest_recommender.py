from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np

# Regiones soportadas para métricas de mercado/transmisiones
REGIONS = ["bcn","cat","esp"]

# Orden estándar de columnas solicitado por negocio
OUTPUT_COLS = [
    "link_ficha",
    "make_clean",
    "modelo_base_x",
    "segmento",
    "year_bca",
    "mileage",
    "fuel_type",
    "transmission",
    "sale_country",
    "sale_name",
    "winning_bid",
    "precio_final_eur",
    "precio_venta_ganvam",
    "margin_abs",
    "vat_type",
    "units_abs_bcn",
    "units_abs_cat",
    "units_abs_esp",
]

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
    # knobs
    alpha_margin: float = 0.6   # margen vs rotación
    rotation_boost: float = 0.0
    demand: DemandConfig = field(default_factory=DemandConfig)

class BCAInvestRecommender:
    """Motor de recomendación y helpers de consultas."""
    def __init__(self, df: pd.DataFrame, cfg: Optional[RecommenderConfig] = None):
        self.df = df.copy()
        self.cfg = cfg or RecommenderConfig()
        self._normalize_types()
        self._check_columns_minimum()
        self._build_price_means()

    # ------------------------- Core prep -------------------------
    def _normalize_types(self):
        # numéricos base
        for c in ["anio","year","mileage","km","kilometros","kilómetros","odometro","odómetro",
                  "precio_final_eur","precio_venta_ganvam","margin_abs","margin_ptc","margin_pct"]:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce")
        # strings frecuentes
        for c in ["make_clean","marca","modelo","model","modelo_base_x","modelo_base_y",
                  "modelo_base_match","model_bca_raw","modelo_detectado"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).str.strip()
        if "combustible_norm" in self.df.columns:
            self.df["combustible_norm"] = self.df["combustible_norm"].astype(str).str.upper().str.strip()
        if "fuel_type" in self.df.columns:
            self.df["fuel_type"] = self.df["fuel_type"].astype(str).str.strip()
        for c in ["sale_name","auction_name","sale_country","transmission","vat_type"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).str.strip()
        # normalizar posibles columnas de URL/IDs
        for c in ["url","link","lote_url","listing_url","link_ficha"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).str.strip()
        for c in ["lot_id","lote_id","listing_id","id_bca"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).str.strip()

    def _check_columns_minimum(self):
        required = [
            # para la base
            "precio_final_eur","precio_venta_ganvam","margin_abs",
            # para clusters/filtrado
            "sale_country",
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

    def _build_price_means(self):
        # Medias de referencia (por si se necesitan)
        keys = [c for c in ["marca","modelo","anio","combustible_norm","sale_country"] if c in self.df.columns]
        if keys and "precio_final_eur" in self.df.columns:
            self.price_mean_country = (
                self.df.groupby(keys, dropna=False)["precio_final_eur"]
                .mean().rename("precio_medio_modelo_pais").reset_index()
            )
        else:
            self.price_mean_country = pd.DataFrame()
        keys2 = [c for c in ["marca","modelo","anio","combustible_norm","sale_country","sale_name"] if c in self.df.columns]
        if keys2 and "precio_final_eur" in self.df.columns:
            self.price_mean_subasta = (
                self.df.groupby(keys2, dropna=False)["precio_final_eur"]
                .mean().rename("precio_medio_modelo_pais_subasta").reset_index()
            )
        else:
            self.price_mean_subasta = pd.DataFrame()

    # ------------------------- Helpers -------------------------
    def _region_field(self, base: str, region: str) -> str:
        return f"{base}_{region}"

    def _fast_rotation_proxy(self, region: str) -> pd.Series:
        # 1) mix 0-3 años (si trae 0-100 convertir a 0-1)
        col_mix = self._region_field("mix_0_3_%", region)
        if col_mix in self.df.columns:
            s = pd.to_numeric(self.df[col_mix], errors="coerce")
            if s.max(skipna=True) and s.max(skipna=True) > 1.0:
                s = s / 100.0
            return s.fillna(0.0)
        # 2) ranking inverso
        col_rank = self._region_field("rank_year_model", region)
        if col_rank in self.df.columns:
            r = pd.to_numeric(self.df[col_rank], errors="coerce")
            return (1.0 / (1.0 + r)).fillna(0.0)
        return pd.Series(0.0, index=self.df.index)

    def _normalize(self, s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            return s.fillna(0.0)
        a = s.min(skipna=True); b = s.max(skipna=True)
        if pd.isna(a) or pd.isna(b) or a==b:
            return s.fillna(0.0)
        return (s - a) / (b - a)

    def _demand_factor(self, region: str) -> pd.Series:
        dc = self.cfg.demand
        parts, weights = [], []
        if dc.use_brand_share:
            col = self._region_field("share_marca_%", region)
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.max(skipna=True) and s.max(skipna=True) > 1.0:
                    s = s / 100.0
                parts.append(self._normalize(s)); weights.append(dc.weight_brand_share)
        if dc.use_units_abs:
            col = self._region_field("units_abs", region)
            if col in self.df.columns:
                parts.append(self._normalize(self.df[col])); weights.append(dc.weight_units_abs)
        if dc.use_concentration_penalty:
            col = self._region_field("dominancia_modelo_marca_%", region)
            if col in self.df.columns:
                s = pd.to_numeric(self.df[col], errors="coerce")
                if s.max(skipna=True) and s.max(skipna=True) > 1.0:
                    s = s / 100.0
                parts.append(1.0 - self._normalize(s)); weights.append(dc.weight_concentration)

        if not parts:
            return pd.Series(1.0, index=self.df.index)  # neutral
        wsum = sum(weights) if sum(weights) != 0 else 1.0
        demand = sum(p*w for p,w in zip(parts, weights)) / wsum
        return demand.clip(0,1).fillna(0.0)

    def _composite_score(self, region: str) -> pd.Series:
        alpha = float(self.cfg.alpha_margin)
        rot_weight = max(0.0, min(1.0, 1.0 - alpha + self.cfg.rotation_boost))
        mar_weight = max(0.0, min(1.0, alpha))

        margin = self._normalize(self.df.get("margin_abs", 0.0))
        demand = self._demand_factor(region)
        margin_demand = margin * demand  # mercado x margen

        rot = self._normalize(self._fast_rotation_proxy(region))

        score = mar_weight * margin_demand + rot_weight * rot
        return score

    def _coalesce(self, row: pd.Series, candidates: List[str], default=np.nan):
        for c in candidates:
            if c in row.index and pd.notna(row[c]) and str(row[c]).strip() != "":
                return row[c]
        return default

    def _compose_transmission_country(self, row: pd.Series) -> str:
        t = str(row.get("transmission","") or "").strip()
        sc = str(row.get("sale_country","") or "").strip()
        if t and sc:
            return f"{t} - {sc}"
        return t or sc or ""

    def _format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conforma el dataframe al layout OUTPUT_COLS (crea/renombra/concatena si hace falta)."""
        out = df.copy()

        # Derivados / mapeos
        if "auction_name" not in out.columns and "sale_name" in out.columns:
            out["auction_name"] = out["sale_name"]

        # year: prefer 'year' > 'anio' > 'Año' > 'year_bca'
        if "year" not in out.columns:
            for c in ["anio","Año","year_bca"]:
                if c in out.columns:
                    out["year"] = out[c]
                    break

        # mileage: prefer 'mileage' > 'km'/'kilometros'/'odometro'
        if "mileage" not in out.columns:
            for c in ["km","kilometros","kilómetros","odometro","odómetro"]:
                if c in out.columns:
                    out["mileage"] = out[c]
                    break

        # fuel_type: prefer 'fuel_type' > 'combustible_norm'
        if "fuel_type" not in out.columns and "combustible_norm" in out.columns:
            out["fuel_type"] = out["combustible_norm"]

        # modelo_base: coalesce varios candidatos
        if "modelo_base_x" not in out.columns:
            for c in ["modelo_base","modelo_base_y","modelo_base_match","modelo"]:
                if c in out.columns:
                    out["modelo_base_x"] = out[c]; break

        # transmission-sale_country combinado
        out["transmission-sale_country"] = out.apply(self._compose_transmission_country, axis=1)

        # Crear faltantes vacíos
        for c in OUTPUT_COLS:
            if c not in out.columns:
                out[c] = np.nan

        # devolver SOLO las columnas previstas y en el orden deseado
        return out[OUTPUT_COLS]

    # ------------------------- API de recomendación -------------------------
    def recommend_best(self,
                       region: str,
                       max_age_years: Optional[int] = None,
                       max_price: Optional[float] = None,
                       min_price: Optional[float] = None,
                       ignore_rotation: bool = False,
                       prefer_fast: bool = False,
                       brand_only: bool = False,
                       # selección del "vehículo óptimo"
                       selection: str = "cheapest",  # por defecto "vehículo barato"=óptimo
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

        # filtros
        if max_age_years is not None and "anio" in df.columns:
            from datetime import datetime
            current_year = datetime.now().year
            min_year = current_year - int(max_age_years)
            df = df[df["anio"] >= min_year]

        if max_price is not None:
            df = df[pd.to_numeric(df["precio_final_eur"], errors="coerce") <= float(max_price)]
        if min_price is not None:
            df = df[pd.to_numeric(df["precio_final_eur"], errors="coerce") >= float(min_price)]

        if year_exact is not None and "anio" in df.columns:
            years = [int(x) for x in (year_exact if isinstance(year_exact,(list,tuple,set)) else [year_exact])]
            df = df[df["anio"].isin(years)]

        seg_col = next((c for c in ["segmento","segment","segmento_norm"] if c in df.columns), None)
        if seg_col:
            if segment_include is not None:
                vals = {str(x).strip().upper() for x in (segment_include if isinstance(segment_include,(list,tuple,set)) else [segment_include])}
                col = (df[seg_col].astype(str).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.upper())
                df = df[col.isin(vals)]
            if segment_exclude is not None:
                vals = {str(x).strip().upper() for x in (segment_exclude if isinstance(segment_exclude,(list,tuple,set)) else [segment_exclude])}
                col = (df[seg_col].astype(str).str.normalize("NFKD").str.encode("ascii","ignore").str.decode("ascii").str.upper())
                df = df[~col.isin(vals)]

        km_col = next((c for c in ["mileage","km","kilometros","kilómetros","odometro","odómetro"] if c in df.columns), None)
        if km_col:
            if mileage_min is not None:
                df = df[pd.to_numeric(df[km_col], errors="coerce") >= float(mileage_min)]
            if mileage_max is not None:
                df = df[pd.to_numeric(df[km_col], errors="coerce") <= float(mileage_max)]

        # ajustar pesos
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

        group_keys = [c for c in ["marca", "modelo_base_x", "modelo", "anio", "combustible_norm"] if c in df.columns]
        if include_sale_country and "sale_country" in df.columns: group_keys.append("sale_country")
        if include_sale_name and "sale_name" in df.columns:       group_keys.append("sale_name")

        if min_listings_per_group > 1 and group_keys:
            sizes = df.groupby(group_keys, dropna=False)["precio_final_eur"].size().rename("n_listings").reset_index()
            df = df.merge(sizes, on=group_keys, how="left")
            df = df[df["n_listings"] >= int(min_listings_per_group)].drop(columns=["n_listings"])

        # Selección "vehículo óptimo" = más barato por grupo con desempate por score
        if selection.lower() == "cheapest":
            df["precio_final_eur"] = pd.to_numeric(df["precio_final_eur"], errors="coerce")
            df = df[df["precio_final_eur"].notna()]
            if df.empty:
                return df.head(0)
            df_sorted = df.sort_values(["precio_final_eur","score"], ascending=[True, False])
            picked = (df_sorted.groupby(group_keys, dropna=False, as_index=False).first()
                      if group_keys else df_sorted.copy())
            # orden final: por score y margen del candidato escogido
            sort_cols = ["score","margin_abs"] if not ignore_rotation else ["margin_abs","score"]
            picked = picked.sort_values(sort_cols, ascending=False)
            return self._format_output(picked.head(n))

        # Modo "mean": medias por cluster
        agg = df.groupby(group_keys, dropna=False).agg(
            precio_medio=("precio_final_eur","mean"),
            margin_abs_medio=("margin_abs","mean"),
            n_listings=("precio_final_eur","size"),
            score=("score","mean")
        ).reset_index()
        sort_cols = ["score","margin_abs_medio"] if not ignore_rotation else ["margin_abs_medio","score"]
        agg = agg.sort_values(sort_cols, ascending=False)
        return self._format_output(agg.head(n))

    # ------------------------- Queries especiales (preguntas) -------------------------
    def q1_best_auction_for_model(self, model_query: str, region: str = "bcn",
                                  top_n: int = 20,
                                  year_from: Optional[int] = None,
                                  year_to: Optional[int] = None,
                                  mileage_max: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Top-N listados para un modelo, con cobertura de años y ranking de subastas.

        Lógica:
        - Filtra el DataFrame por modelo (cadenas que contienen `model_query`).
        - Aplica filtros de año (por defecto [2018, 2024]) y kilometraje máximo.
        - Score Q1 basado SOLO en margin_abs:

              score_q1 = margin_abs_normalizado

          (la normalización no cambia el orden, es solo por comodidad numérica).
        - Cobertura de años: para cada año del intervalo, selecciona el mejor
          listado (mayor score_q1) de ese año. Si el número de años es menor
          que `top_n`, rellena el resto con los siguientes mejores listados por
          score_q1, sin volver a repetir el mismo registro.
        - Devuelve:
          * top_listings: listado ordenado por score_q1 descendente.
          * rank_subastas: ranking de subastas (sale_name) dentro de ese Top-N.
        """
        df = self.df.copy()

        # 1) Filtro por modelo usando modelo_base_x/modelo/model...
        mask_parts = []
        for col in ["modelo_base_x", "modelo", "model", "model_bca_raw", "modelo_detectado"]:
            if col in df.columns:
                mask_parts.append(
                    df[col].astype(str).str.contains(model_query, case=False, na=False)
                )

        if mask_parts:
            mask_model = mask_parts[0]
            for m in mask_parts[1:]:
                mask_model = mask_model | m
        else:
            mask_model = pd.Series(False, index=df.index)

        df = df[mask_model]

        # 2) Filtro por años (por defecto 2018-2024)
        if year_from is None:
            year_from = 2018
        if year_to is None:
            year_to = 2024

        if "anio" in df.columns:
            df = df[(df["anio"] >= int(year_from)) & (df["anio"] <= int(year_to))]

        # 3) Filtro por kilometraje
        if mileage_max is not None:
            km_col = next((c for c in ["mileage","km","kilometros","kilómetros","odometro","odómetro"]
                           if c in df.columns), None)
            if km_col:
                df = df[pd.to_numeric(df[km_col], errors="coerce") <= float(mileage_max)]

        if df.empty:
            return df.head(0), pd.DataFrame(columns=["sale_name","auction_score","representation_pct","price_adv_pct"])

        # 4) Score Q1 basado SOLO en margen absoluto (normalizado)
        margin = pd.to_numeric(df.get("margin_abs", 0.0), errors="coerce")

        def _norm(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() == 0:
                return s.fillna(0.0)
            a, b = s.min(), s.max()
            if a == b:
                return s.fillna(0.0)
            return (s - a) / (b - a)

        margin_n = _norm(margin)
        df = df.assign(score_q1=margin_n)

        # 5) Cobertura de años: 1 por año del intervalo primero, luego rellenar hasta top_n
        years = sorted(df.get("anio", pd.Series(dtype=float)).dropna().unique())
        selected_idx = []

        for y in years:
            year_mask = df["anio"] == y
            cand = df[year_mask]
            if cand.empty:
                continue
            best_row = cand.sort_values("score_q1", ascending=False).iloc[0]
            selected_idx.append(best_row.name)
            if len(selected_idx) >= top_n:
                break

        # Si aún no alcanzamos top_n, rellenar con mejores restantes (sea del año que sea)
        if len(selected_idx) < top_n:
            remaining = df.drop(index=selected_idx, errors="ignore")
            remaining_sorted = remaining.sort_values("score_q1", ascending=False)
            extra_idx = list(remaining_sorted.index[: max(0, top_n - len(selected_idx))])
            selected_idx.extend(extra_idx)

        df_top = df.loc[selected_idx]
        # Orden final por score_q1 descendente (equivalente a ordenar por margin_abs)
        df_top = df_top.sort_values("score_q1", ascending=False)

        # 6) Preparar top_listings con formato estándar
        top_listings = self._format_output(df_top)

        # 7) Ranking de subastas (sale_name) basado en representación y ventaja de precio
        if top_listings.empty:
            return top_listings, pd.DataFrame(columns=["sale_name","auction_score","representation_pct","price_adv_pct"])

        # Representation
        rep = (
            top_listings.groupby("sale_name").size().rename("n").reset_index()
            .assign(representation_pct=lambda d: 100.0 * d["n"] / float(len(top_listings)))
        )

        # Price advantage: comparando precio_final_eur de cada listing contra la mediana
        base_cols = [c for c in ["marca","modelo","anio","combustible_norm"] if c in self.df.columns]
        enriched = top_listings.copy()
        if base_cols and "precio_final_eur" in self.df.columns:
            # merge por link_ficha + sale_name + winning_bid si es posible
            keys = [c for c in ["link_ficha","sale_name","winning_bid"]
                    if c in top_listings.columns and c in self.df.columns]
            if keys:
                enriched = top_listings.merge(
                    self.df[base_cols + keys + ["precio_final_eur"]],
                    on=keys,
                    how="left",
                )
            elif "link_ficha" in top_listings.columns and "link_ficha" in self.df.columns:
                enriched = top_listings.merge(
                    self.df[base_cols + ["link_ficha","precio_final_eur"]],
                    on=["link_ficha"],
                    how="left",
                )

        if base_cols and "precio_final_eur" in enriched.columns:
            med = (
                self.df.groupby(base_cols)["precio_final_eur"].median()
                .rename("precio_median_cluster").reset_index()
            )
            enriched = enriched.merge(med, on=base_cols, how="left")
            enriched["price_adv_pct"] = 100.0 * (
                enriched["precio_median_cluster"] - enriched["precio_final_eur"]
            ) / enriched["precio_median_cluster"]
        else:
            enriched["price_adv_pct"] = 0.0

        adv = enriched.groupby("sale_name")["price_adv_pct"].mean().reset_index()

        # score compuesto (60% representación, 40% ventaja precio normalizada)
        def _norm2(s):
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() == 0:
                return s.fillna(0.0)
            a, b = s.min(), s.max()
            return s.fillna(0.0) if a == b else (s - a) / (b - a)

        repn = _norm2(rep["representation_pct"]).rename("repn")
        advn = _norm2(adv["price_adv_pct"]).rename("advn")
        rank = rep.assign(_repn=repn.values).merge(
            adv.assign(_advn=advn.values), on="sale_name"
        )
        rank["auction_score"] = 0.6 * rank["_repn"] + 0.4 * rank["_advn"]
        rank = rank[
            ["sale_name", "auction_score", "representation_pct", "price_adv_pct"]
        ].sort_values("auction_score", ascending=False)

        return top_listings, rank



    def q2_price_order_within_brand(self, brand: str, region: str="bcn",
                                    min_year: int = 2020,
                                    max_km: float = 100000.0,
                                    mode: str = "cheapest") -> pd.DataFrame:
        """Dentro de una marca: un solo vehículo por modelo_base_x.

        - Filtra por marca (columna 'marca').
        - Aplica filtros estándar: año >= min_year, km <= max_km (configurables).
        - Agrupa por modelo_base_x (o equivalente) y selecciona:
            * mode="cheapest"   -> la unidad con menor precio_final_eur.
            * mode="max_margin" -> la unidad con mayor margin_abs.
        - Ordena el resultado:
            * cheapest   -> por precio_final_eur ascendente.
            * max_margin -> por margin_abs descendente.
        """
        df = self.df.copy()

        # 1) Filtro por marca
        if "marca" in df.columns:
            df = df[df["marca"].astype(str).str.contains(brand, case=False, na=False)]

        # 2) Filtro por año mínimo
        if min_year is not None and "anio" in df.columns:
            df = df[pd.to_numeric(df["anio"], errors="coerce") >= int(min_year)]

        # 3) Filtro por km máximo
        if max_km is not None:
            km_col = next((c for c in ["mileage","km","kilometros","kilómetros","odometro","odómetro"]
                           if c in df.columns), None)
            if km_col:
                df = df[pd.to_numeric(df[km_col], errors="coerce") <= float(max_km)]

        if df.empty:
            return self._format_output(df).head(0)

        # 4) Columna de modelo base
        model_col = "modelo_base_x"
        if model_col not in df.columns:
            for c in ["modelo_base","modelo_base_y","modelo_base_match","modelo"]:
                if c in df.columns:
                    model_col = c
                    break

        # 5) Asegurar numéricos
        df["precio_final_eur"] = pd.to_numeric(df.get("precio_final_eur", 0.0), errors="coerce")
        df["margin_abs"] = pd.to_numeric(df.get("margin_abs", 0.0), errors="coerce")

        g = df.groupby(model_col, dropna=False)

        mode_l = (mode or "cheapest").lower()
        if mode_l == "max_margin":
            idx = g["margin_abs"].idxmax()
        else:
            # por defecto cheapest
            idx = g["precio_final_eur"].idxmin()

        df_sel = df.loc[idx].copy()

        # 6) Orden final
        if mode_l == "max_margin":
            df_sel = df_sel.sort_values("margin_abs", ascending=False)
        else:
            df_sel = df_sel.sort_values("precio_final_eur", ascending=True)

        # 7) Formatear salida al layout estándar (modelo_base_x, year_bca, etc.)
        return self._format_output(df_sel).reset_index(drop=True)


    def q3_price_order_within_segment(self,
                                      segment: str,
                                      region: str = "bcn",
                                      top_n: int = 20,
                                      year_from: Optional[int] = None,
                                      year_to: Optional[int] = None,
                                      km_max: Optional[float] = None,
                                      fuel_include: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """
        Dentro de un segmento: 1) filtra, 2) elige el más barato por cluster (marca+modelo_base+anio+fuel),
        3) ordena por margin_abs desc y devuelve Top-N sin repetir cluster.
        Defaults: 2020–2025, 100.000 km, fuel opcional.
        """
        df = self.df.copy()

        # --- segment filter (robusto) ---
        seg_col = next((c for c in ["segmento","segment","segmento_norm"] if c in df.columns), None)
        if not seg_col:
            raise ValueError("No existe columna segmento/segment/segmento_norm en el dataset")
        seg_val = str(segment).strip().upper()
        df = df[df[seg_col].astype(str).str.upper() == seg_val]

        # --- defaults ---
        if year_from is None: year_from = 2020
        if year_to   is None: year_to   = 2025
        if km_max    is None: km_max    = 100000

        # --- year filter ---
        if "anio" in df.columns:
            df = df[(pd.to_numeric(df["anio"], errors="coerce") >= int(year_from)) &
                    (pd.to_numeric(df["anio"], errors="coerce") <= int(year_to))]

        # --- km filter ---
        km_col = next((c for c in ["mileage","km","kilometros","kilómetros","odometro","odómetro"] if c in df.columns), None)
        if km_col is not None and km_max is not None:
            df = df[pd.to_numeric(df[km_col], errors="coerce") <= float(km_max)]

        # --- fuel filter (opcional) ---
        if fuel_include is not None and "combustible_norm" in df.columns:
            fuels = {str(x).strip().upper() for x in (fuel_include if isinstance(fuel_include,(list,tuple,set)) else [fuel_include])}
            df = df[df["combustible_norm"].astype(str).str.upper().isin(fuels)]

        if df.empty:
            return self._format_output(df).head(0)

        # --- score informativo (no decide el ranking final) ---
        prev_cfg = self.cfg
        self.cfg = RecommenderConfig(alpha_margin=prev_cfg.alpha_margin,
                                     rotation_boost=prev_cfg.rotation_boost,
                                     demand=prev_cfg.demand)
        df = df.copy()
        df["score"] = self._composite_score(region).loc[df.index].fillna(0.0)
        self.cfg = prev_cfg

        # --- cheapest por cluster (sin país/subasta) ---
        model_col = "modelo_base_x" if "modelo_base_x" in df.columns else \
                    next((c for c in ["modelo_base","modelo_base_y","modelo_base_match","modelo"] if c in df.columns), "modelo")
        group_keys = ["marca", model_col, "anio", "combustible_norm"]
        for c in ["precio_final_eur","margin_abs"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # orden interno para el pick: precio asc, desempate por mayor margen y mayor score
        df_sorted = df.sort_values(["precio_final_eur","margin_abs","score"], ascending=[True, False, False])
        cheapest_per_cluster = df_sorted.groupby(group_keys, dropna=False, as_index=False).first()

        # --- Top-N por margen (desc), desempate por score ---
        topn = cheapest_per_cluster.sort_values(["margin_abs","score"], ascending=[False, False]).head(int(top_n))

        # --- salida en layout estándar ---
        return self._format_output(topn).reset_index(drop=True)


    def q5_best_fuel_gap(self, modelo_base: str, anio: int) -> pd.DataFrame:
        """Mejor fuel por región y gap vs resto."""
        df = self.df.copy()

        # --- Filtro robusto por modelo_base_x / modelo_base / modelo y año ---
        model_masks = []
        for col in ["modelo_base_x", "modelo_base", "modelo"]:
            if col in df.columns:
                model_masks.append(
                    df[col].astype(str).str.contains(modelo_base, case=False, na=False)
                )

        if model_masks:
            mask_model = model_masks[0]
            for m in model_masks[1:]:
                mask_model = mask_model | m
        else:
            mask_model = pd.Series(False, index=df.index)

        mask = (df.get("anio", np.nan) == int(anio)) & mask_model
        sub = df[mask]

        out_rows = []
        for r in REGIONS:
            best_col = f"best_fuel_{r}"
            gap_col  = f"row_vs_best_fuel_%_{r}"

            best = sub.get(best_col)
            gap  = sub.get(gap_col)

            if best is None and gap is None:
                continue

            # mejor fuel: modo de best_fuel_{r} si existe; si no, fuel_type más frecuente
            if best is not None and best.notna().any():
                best_mode = best.dropna().astype(str).str.upper().mode()
                best_val = best_mode.iloc[0] if not best_mode.empty else np.nan
            else:
                fuel_series = sub.get("fuel_type", pd.Series(dtype=str)).astype(str).str.upper()
                best_mode = fuel_series.mode()
                best_val = best_mode.iloc[0] if not best_mode.empty else np.nan

            # gap medio: 1 - media(row_vs_best...), ajustando % si hace falta
            if gap is not None and gap.notna().any():
                g = pd.to_numeric(gap, errors="coerce")
                if g.max(skipna=True) and g.max(skipna=True) > 1.0:
                    g = g / 100.0
                gap_mean = float(1.0 - np.nanmean(g))
            else:
                gap_mean = np.nan

            out_rows.append({"region": r, "best_fuel": best_val, "gap_vs_rest_mean": gap_mean})

        return pd.DataFrame(out_rows)


# ------------------------- Carga de dataset -------------------------
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
