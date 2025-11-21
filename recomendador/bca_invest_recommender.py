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
    "combustible_norm",
    "sale_name",
    "winning_bid",
    "precio_final_eur",
    "precio_venta_ganvam",
    "margin_abs",
    "vat_type",
    "units_abs_bcn",
    "units_abs_cat",
    "units_abs_esp",
    "YoY_weighted_esp",
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
        # 2) Alinear a self.df.index
        aligned = [p.reindex(self.df.index).astype(float) for p in parts]

        num = 0.0
        den = 0.0
        for s, w in zip(aligned, weights):
            mask = s.notna()
            num = num + s.fillna(0.0) * (w * mask)
            den = den + (w * mask)

        den = den.replace(0, np.nan)
        demand = num / den

        # Fila sin ningún sensor → valor neutro (0.5) en vez de 0
        return demand.fillna(0.5).clip(0, 1)

    def _composite_score(self, region: str) -> pd.Series:
        alpha = float(self.cfg.alpha_margin)

        base_rot = max(0.0, 1.0 - alpha)
        rot_boost = max(0.0, float(self.cfg.rotation_boost))

        rot_raw = base_rot + rot_boost
        mar_raw = max(0.0, alpha)

        total = mar_raw + rot_raw
        if total > 0:
            mar_weight = mar_raw / total
            rot_weight = rot_raw / total
        else:
            mar_weight = rot_weight = 0.5

        margin = self._normalize(self.df.get("margin_abs", 0.0))
        demand = self._demand_factor(region)
        margin_demand = margin * demand

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

        # auction_name a partir de sale_name si hace falta (por si lo usas en algún sitio)
        if "auction_name" not in out.columns and "sale_name" in out.columns:
            out["auction_name"] = out["sale_name"]

        # year: prefer 'year' > 'anio' > 'Año' > 'year_bca'
        if "year" not in out.columns:
            for c in ["anio","Año","year_bca"]:
                if c in out.columns:
                    out["year"] = out[c]
                    break

        # mileage
        if "mileage" not in out.columns:
            for c in ["km","kilometros","kilómetros","odometro","odómetro"]:
                if c in out.columns:
                    out["mileage"] = out[c]
                    break

        # fuel normalizado en combustible_norm si falta
        if "combustible_norm" not in out.columns and "fuel_type" in out.columns:
            out["combustible_norm"] = out["fuel_type"]

        # modelo_base_x
        if "modelo_base_x" not in out.columns:
            for c in ["modelo_base","modelo_base_y","modelo_base_match","modelo"]:
                if c in out.columns:
                    out["modelo_base_x"] = out[c]
                    break

        # OJO: ya NO creamos transmission-sale_country aquí

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

        # ----------------------------------------------------
        # Selección según 'selection': cheapest / max_margin / mean
        # ----------------------------------------------------
        sel = (selection or "cheapest").lower()

        # Normalizamos numéricos que vamos a usar
        if "precio_final_eur" in df.columns:
            df["precio_final_eur"] = pd.to_numeric(df["precio_final_eur"], errors="coerce")
        if "margin_abs" in df.columns:
            df["margin_abs"] = pd.to_numeric(df["margin_abs"], errors="coerce")

        # --- Modo "cheapest": vehículo más barato por cluster ---
        if sel == "cheapest":
            df_cheapest = df[df["precio_final_eur"].notna()]
            if df_cheapest.empty:
                return self._format_output(df_cheapest.head(0))

            # orden interno: precio asc, desempate por score desc
            df_sorted = df_cheapest.sort_values(
                ["precio_final_eur", "score"], ascending=[True, False]
            )

            # un vehículo por cluster (el primero tras ese orden)
            picked = (
                df_sorted.groupby(group_keys, dropna=False, as_index=False).first()
                if group_keys else df_sorted.copy()
            )

            # orden final de los clusters: por score y margen
            sort_cols = ["score", "margin_abs"] if not ignore_rotation else ["margin_abs", "score"]
            picked = picked.sort_values(sort_cols, ascending=False)

            return self._format_output(picked.head(n))

        # --- Modo "max_margin": vehículo con mayor margen por cluster ---
        elif sel == "max_margin":
            df_mm = df[df["margin_abs"].notna()]
            if df_mm.empty:
                return self._format_output(df_mm.head(0))

            # orden interno: margen desc, desempate por score desc
            df_sorted = df_mm.sort_values(
                ["margin_abs", "score"], ascending=[False, False]
            )

            # un vehículo por cluster (el primero tras ese orden)
            picked = (
                df_sorted.groupby(group_keys, dropna=False, as_index=False).first()
                if group_keys else df_sorted.copy()
            )

            # orden final de los clusters: margen desc, desempate por score
            picked = picked.sort_values(["margin_abs", "score"], ascending=[False, False])

            return self._format_output(picked.head(n))

        # --- Modo "mean" (u otros): medias por cluster ---
        else:
            if not group_keys:
                # sin claves de cluster, devolvemos el df ordenado por score/margen
                sort_cols = ["score", "margin_abs"] if not ignore_rotation else ["margin_abs", "score"]
                df_sorted = df.sort_values(sort_cols, ascending=False)
                return self._format_output(df_sorted.head(n))

            agg = df.groupby(group_keys, dropna=False).agg(
                precio_medio=("precio_final_eur", "mean"),
                margin_abs_medio=("margin_abs", "mean"),
                n_listings=("precio_final_eur", "size"),
                score=("score", "mean"),
            ).reset_index()

            # orden de clusters: por score y margen medio
            sort_cols = ["score", "margin_abs_medio"] if not ignore_rotation else ["margin_abs_medio", "score"]
            agg = agg.sort_values(sort_cols, ascending=False)

            # IMPORTANTE: _format_output espera nombres como precio_final_eur / margin_abs,
            # así que renombramos si quieres que salga con el layout estándar.
            agg = agg.rename(
                columns={
                    "precio_medio": "precio_final_eur",
                    "margin_abs_medio": "margin_abs",
                }
            )

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

        # 1) Filtro por modelo SOLO usando modelo_base_x
        if "modelo_base_x" not in df.columns:
            raise ValueError(
                "q1_best_auction_for_model requiere la columna 'modelo_base_x' para filtrar el modelo."
            )

        # Versión con 'contiene' (X1 también captura IX1):
        mask_model = df["modelo_base_x"].astype(str).str.contains(
            model_query, case=False, na=False
        )

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


    def q3_price_order_within_segment(
        self,
        segment: str,
        region: str = "bcn",
        top_n: int = 20,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        km_max: Optional[float] = None,
        fuel_include: Optional[Union[str, List[str]]] = None,
        mode: str = "cheapest",
    ) -> pd.DataFrame:
        """
        Q3 — Dentro de un segmento, ¿cuáles son los mejores lotes?

        Lógica:
        1) Filtra por segmento + rango de años + km máximos + fuel opcional.
        2) Calcula un score informativo usando el modelo de recomendación.
        3) Ordena:
           - mode="cheapest"   -> precio_final_eur asc, desempate por margin_abs desc y score desc.
           - mode="max_margin" -> margin_abs desc, desempate por score desc y precio_final_eur asc.
        4) Aplica un límite de repetición por cluster (marca+modelo+anio+fuel):
           máximo el 10% de N por cluster.
        5) Devuelve los Top-N en el layout estándar.
        """
        df = self.df.copy()

        # --- segment filter (robusto) ---
        seg_val = str(segment).strip().upper()
        seg_col = None
        for c in ["segmento_norm", "segmento", "segment"]:
            if c in df.columns:
                seg_col = c
                break
        if seg_col is None:
            raise ValueError(
                "No se encuentra ninguna columna de segmento "
                "('segmento_norm'/'segmento'/'segment')."
            )

        df = df[df[seg_col].astype(str).str.upper() == seg_val]

        # --- defaults ---
        if year_from is None:
            year_from = 2020
        if year_to is None:
            year_to = 2025
        if km_max is None:
            km_max = 100000

        # --- year filter ---
        year_col = "anio" if "anio" in df.columns else (
            "year" if "year" in df.columns else None
        )
        if year_col is not None:
            year_num = pd.to_numeric(df[year_col], errors="coerce")
            df = df[(year_num >= int(year_from)) & (year_num <= int(year_to))]

        # --- km filter ---
        km_col = None
        for c in ["mileage", "km", "kilometros", "kilómetros", "odometro", "odómetro"]:
            if c in df.columns:
                km_col = c
                break
        if km_col is not None and km_max is not None:
            df = df[pd.to_numeric(df[km_col], errors="coerce") <= float(km_max)]

        # --- fuel filter (opcional) ---
        if fuel_include is not None and "combustible_norm" in df.columns:
            if isinstance(fuel_include, (list, tuple, set)):
                fuels = {str(x).strip().upper() for x in fuel_include}
            else:
                fuels = {str(fuel_include).strip().upper()}
            df = df[df["combustible_norm"].astype(str).str.upper().isin(fuels)]

        if df.empty:
            return self._format_output(df).head(0)

        # --- score informativo (no decide el ranking principal) ---
        prev_cfg = self.cfg
        try:
            self.cfg = RecommenderConfig(
                alpha_margin=prev_cfg.alpha_margin,
                rotation_boost=prev_cfg.rotation_boost,
                demand=prev_cfg.demand,
            )
            score_series = self._composite_score(region)
            df = df.copy()
            df["score"] = score_series.loc[df.index].fillna(0.0)
        finally:
            self.cfg = prev_cfg

        # asegurar numéricos
        for c in ["precio_final_eur", "margin_abs"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # --- orden principal según mode ---
        mode_norm = (mode or "cheapest").lower()
        if mode_norm not in {"cheapest", "max_margin"}:
            mode_norm = "cheapest"

        if mode_norm == "cheapest":
            # precio asc, desempate por margen y score
            sort_cols = ["precio_final_eur", "margin_abs", "score"]
            ascending = [True, False, False]
        else:  # max_margin
            # margen desc, desempate por score y precio
            sort_cols = ["margin_abs", "score", "precio_final_eur"]
            ascending = [False, False, True]

        df_sorted = df.sort_values(sort_cols, ascending=ascending)

        # --- límite de repetición por cluster (máx 10% de N) ---
        N = int(top_n)
        if N <= 0:
            return self._format_output(df_sorted.head(0))

        max_per_cluster = int(N * 0.10)  # 10% de N
        if max_per_cluster < 1:
            max_per_cluster = 1

        # Definimos el "cluster" como marca+modelo+anio+combustible
        brand_col = None
        for c in ["make_clean", "marca"]:
            if c in df_sorted.columns:
                brand_col = c
                break
        model_col = None
        for c in ["modelo_base_x", "modelo_base", "modelo"]:
            if c in df_sorted.columns:
                model_col = c
                break
        year_col = "anio" if "anio" in df_sorted.columns else (
            "year" if "year" in df_sorted.columns else None
        )
        fuel_col = "combustible_norm" if "combustible_norm" in df_sorted.columns else None

        def cluster_key(row):
            key_parts = []
            for col in [brand_col, model_col, year_col, fuel_col]:
                if col is not None:
                    key_parts.append(str(row.get(col, "")))
            return tuple(key_parts)

        selected_idx: List[Any] = []
        counts: Dict[tuple, int] = {}

        for idx, row in df_sorted.iterrows():
            key = cluster_key(row)
            current = counts.get(key, 0)
            if current >= max_per_cluster:
                # ya hemos sacado el 10% de N de este cluster
                continue
            selected_idx.append(idx)
            counts[key] = current + 1
            if len(selected_idx) >= N:
                break

        df_out = df_sorted.loc[selected_idx]
        return self._format_output(df_out).reset_index(drop=True)

def q4_attractiveness_for_vehicle(
    self,
    modelo_base: str,
    region: str = "bcn",
    year: int | None = None,
    fuel: str | list[str] | None = None,
    year_from: int = 2020,
    year_to: int = 2024,
) -> pd.DataFrame:
    """
    Evalúa qué tan atractivo es un modelo_base_x (+año+fuel) en una región.

    Devuelve una fila con:
      - score medio,
      - atractivo en %,
      - bucket BAJO/MEDIO/ALTO,
      - demanda media y rotación media como explicación.
    """
    df = self.df.copy()

    # --- Filtro SOLO por modelo_base_x ---
    if "modelo_base_x" not in df.columns:
        raise ValueError(
            "q4_attractiveness_for_vehicle requiere la columna 'modelo_base_x' para filtrar el modelo."
        )

    mask_model = df["modelo_base_x"].astype(str).str.contains(
        modelo_base, case=False, na=False
    )
    # Igualdad exacta alternativa:
    # mask_model = df["modelo_base_x"].astype(str).str.upper().eq(str(modelo_base).upper())

    df = df[mask_model]

    # --- Filtro de año: exacto si lo pasas, rango 2020–2024 por defecto ---
    if "anio" in df.columns:
        df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
        if year is not None:
            df = df[df["anio"] == int(year)]
        else:
            df = df[(df["anio"] >= int(year_from)) & (df["anio"] <= int(year_to))]

    # --- Filtro de fuel (opcional) ---
    fuels = None
    if fuel is not None:
        fuels = fuel if isinstance(fuel, (list, tuple, set)) else [fuel]
        fuels = {str(f).strip().upper() for f in fuels}
        fuel_col = (
            "combustible_norm"
            if "combustible_norm" in df.columns
            else "fuel_type"
            if "fuel_type" in df.columns
            else None
        )
        if fuel_col:
            df = df[df[fuel_col].astype(str).str.upper().isin(fuels)]

    if df.empty:
        return pd.DataFrame(
            [
                {
                    "region": region,
                    "modelo_base_x": modelo_base,
                    "anio": int(year) if year is not None else None,
                    "fuel": ",".join(sorted(fuels)) if fuels else None,
                    "n_lotes": 0,
                    "score_mean": np.nan,
                    "attractiveness_pct": np.nan,
                    "attractiveness_bucket": "SIN_DATOS",
                    "demand_factor_mean": np.nan,
                    "rotation_proxy_mean": np.nan,
                }
            ]
        )

    # --- Score compuesto global, restringido al subset ---
    score_all = self._composite_score(region)
    score_sub = score_all.loc[df.index].fillna(0.0)

    attractiveness = float(score_sub.mean())  # ~0..1
    attractiveness_pct = round(100.0 * attractiveness, 1)

    # Buckets heurísticos
    if attractiveness < 0.4:
        bucket = "BAJO"
    elif attractiveness < 0.7:
        bucket = "MEDIO"
    else:
        bucket = "ALTO"

    rot_proxy = self._fast_rotation_proxy(region).loc[df.index]
    demand = self._demand_factor(region).loc[df.index]

    return pd.DataFrame(
        [
            {
                "region": region,
                "modelo_base_x": str(modelo_base),
                "anio": int(year) if year is not None else None,
                "fuel": ",".join(
                    sorted(
                        df.get(
                            "combustible_norm",
                            df.get("fuel_type", pd.Series(dtype=str)),
                        )
                        .astype(str)
                        .str.upper()
                        .unique()
                    )
                )
                if ("combustible_norm" in df.columns or "fuel_type" in df.columns)
                else None,
                "n_lotes": int(len(df)),
                "score_mean": attractiveness,
                "attractiveness_pct": attractiveness_pct,
                "attractiveness_bucket": bucket,
                "demand_factor_mean": float(demand.mean()),
                "rotation_proxy_mean": float(rot_proxy.mean()),
            }
        ]
    )
    def q4_attractiveness_for_vehicle(
        self,
        modelo_base: str,
        region: str = "bcn",
        year: int | None = None,
        fuel: str | list[str] | None = None,
        year_from: int = 2020,
        year_to: int = 2024,
    ) -> pd.DataFrame:
        """
        Evalúa qué tan atractivo es un modelo_base_x (+año+fuel) en una región.

        Devuelve una fila con:
          - score medio,
          - atractivo en %,
          - bucket BAJO/MEDIO/ALTO,
          - demanda media y rotación media como explicación.
        """
        df = self.df.copy()

        # --- Filtro SOLO por modelo_base_x ---
        if "modelo_base_x" not in df.columns:
            raise ValueError(
                "q4_attractiveness_for_vehicle requiere la columna 'modelo_base_x' para filtrar el modelo."
            )

        mask_model = df["modelo_base_x"].astype(str).str.contains(
            modelo_base, case=False, na=False
        )
        # Igualdad exacta alternativa:
        # mask_model = df["modelo_base_x"].astype(str).str.upper().eq(str(modelo_base).upper())

        df = df[mask_model]

        # --- Filtro de año: exacto si lo pasas, rango 2020–2024 por defecto ---
        if "anio" in df.columns:
            df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
            if year is not None:
                df = df[df["anio"] == int(year)]
            else:
                df = df[(df["anio"] >= int(year_from)) & (df["anio"] <= int(year_to))]

        # --- Filtro de fuel (opcional) ---
        fuels = None
        if fuel is not None:
            fuels = fuel if isinstance(fuel, (list, tuple, set)) else [fuel]
            fuels = {str(f).strip().upper() for f in fuels}
            fuel_col = (
                "combustible_norm"
                if "combustible_norm" in df.columns
                else "fuel_type"
                if "fuel_type" in df.columns
                else None
            )
            if fuel_col:
                df = df[df[fuel_col].astype(str).str.upper().isin(fuels)]

        if df.empty:
            return pd.DataFrame(
                [
                    {
                        "region": region,
                        "modelo_base_x": modelo_base,
                        "anio": int(year) if year is not None else None,
                        "fuel": ",".join(sorted(fuels)) if fuels else None,
                        "n_lotes": 0,
                        "score_mean": np.nan,
                        "attractiveness_pct": np.nan,
                        "attractiveness_bucket": "SIN_DATOS",
                        "demand_factor_mean": np.nan,
                        "rotation_proxy_mean": np.nan,
                    }
                ]
            )

        # --- Score compuesto global, restringido al subset ---
        score_all = self._composite_score(region)
        score_sub = score_all.loc[df.index].fillna(0.0)

        attractiveness = float(score_sub.mean())  # ~0..1
        attractiveness_pct = round(100.0 * attractiveness, 1)

        # Buckets heurísticos
        if attractiveness < 0.4:
            bucket = "BAJO"
        elif attractiveness < 0.7:
            bucket = "MEDIO"
        else:
            bucket = "ALTO"

        rot_proxy = self._fast_rotation_proxy(region).loc[df.index]
        demand = self._demand_factor(region).loc[df.index]

        return pd.DataFrame(
            [
                {
                    "region": region,
                    "modelo_base_x": str(modelo_base),
                    "anio": int(year) if year is not None else None,
                    "fuel": ",".join(
                        sorted(
                            df.get(
                                "combustible_norm",
                                df.get("fuel_type", pd.Series(dtype=str)),
                            )
                            .astype(str)
                            .str.upper()
                            .unique()
                        )
                    )
                    if ("combustible_norm" in df.columns or "fuel_type" in df.columns)
                    else None,
                    "n_lotes": int(len(df)),
                    "score_mean": attractiveness,
                    "attractiveness_pct": attractiveness_pct,
                    "attractiveness_bucket": bucket,
                    "demand_factor_mean": float(demand.mean()),
                    "rotation_proxy_mean": float(rot_proxy.mean()),
                }
            ]
        )

    def q5_best_fuel_gap(self, modelo_base: str, anio: int) -> pd.DataFrame:
        """Mejor fuel por región y gap vs resto."""
        df = self.df.copy()

        # --- Filtro SOLO por modelo_base_x ---
        if "modelo_base_x" not in df.columns:
            raise ValueError(
                "q5_best_fuel_gap requiere la columna 'modelo_base_x' para filtrar el modelo."
            )

        # Si quieres que "X1" incluya también "IX1", usamos contains:
        mask_model = df["modelo_base_x"].astype(str).str.contains(
            modelo_base, case=False, na=False
        )
        # 👉 Si en algún momento prefieres igualdad exacta (X1 NO incluye IX1):
        # mask_model = df["modelo_base_x"].astype(str).str.upper().eq(str(modelo_base).upper())

        # --- Filtro por año ---
        if "anio" in df.columns:
            year_col = "anio"
        elif "year" in df.columns:
            year_col = "year"
        else:
            year_col = None

        if year_col is not None:
            year_num = pd.to_numeric(df[year_col], errors="coerce")
            mask_year = (year_num == int(anio))
        else:
            # si no hay columna de año, no filtramos por año
            mask_year = pd.Series(True, index=df.index)

        sub = df[mask_model & mask_year]

        out_rows = []
        for r in REGIONS:
            best_col = f"best_fuel_{r}"
            gap_col  = f"row_vs_best_fuel_%_{r}"

            best = sub.get(best_col)
            gap  = sub.get(gap_col)

            if best is None and gap is None:
                continue

            # mejor fuel: modo de best_fuel_{r} si existe; si no, fuel_type/combustible_norm más frecuente
            if best is not None and best.notna().any():
                best_mode = best.dropna().astype(str).str.upper().mode()
                best_val = best_mode.iloc[0] if not best_mode.empty else np.nan
            else:
                fuel_series = sub.get(
                    "combustible_norm",
                    sub.get("fuel_type", pd.Series(dtype=str))
                ).astype(str).str.upper()
                best_mode = fuel_series.mode()
                best_val = best_mode.iloc[0] if not best_mode.empty else np.nan

            # gap medio: 1 - media(row_vs_best...), ajustando % si viene en 0–100
            if gap is not None and gap.notna().any():
                g = pd.to_numeric(gap, errors="coerce")
                if g.max(skipna=True) and g.max(skipna=True) > 1.0:
                    g = g / 100.0
                gap_mean = float(1.0 - np.nanmean(g))
            else:
                gap_mean = np.nan

            out_rows.append(
                {"region": r, "best_fuel": best_val, "gap_vs_rest_mean": gap_mean}
            )

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
