# Recomendador BCA+DGT — Fase 1 (YAML + Runner)

Este paquete te permite **definir preguntas** sobre la base enriquecida BCA+DGT en un **archivo YAML** y ejecutar un **script** que devuelve **CSVs** con los resultados (Top-N, rankings por marca, países/subastas más baratas para un modelo concreto, etc.).

## Archivos

- `bca_invest_recommender.py` — motor de scoring y agregación.
- `run_queries.py` — ejecuta consultas definidas en YAML y guarda resultados en CSV.
- `queries_examples.yaml` — plantilla editable con parámetros y ejemplos (Q1–Q5).

## Requisitos de datos mínimos

Columnas necesarias en el dataset:  
`marca, modelo, anio, combustible_norm, precio_final_eur, precio_venta_ganvam, margin_abs, sale_country, sale_name`

**Recomendadas** (mejoran el ranking):  
- Rotación/DGT: `mix_0_3_%_{region}` o `rank_year_model_{region}` (con `{region}` en `bcn|cat|esp`).  
- Demanda: `units_abs_{region}`, `share_marca_%_{region}`, `dominancia_modelo_marca_%_{region}`.  
- Calidad: `lot_status`, `year_bca`, `anio_ganvam`.  
- Opcional: `margin_ptc` (si existe, se informa como media).

> **Nota regiones:** Actualmente la granularidad es `bcn`, `cat`, `esp`. Para Barcelona/Girona usamos `bcn`/`cat` respectivamente.

## Cómo ejecutar

```bash
python run_queries.py --data "bca_enriched_final.xlsx" --yaml "queries_examples.yaml" --outdir "salida"

## Estructura del YAML

### Cabecera global

```yaml
version: 1
alpha: 0.6  # peso del margen en el score (0..1). La rotación pesa (1 - alpha).

global_filters:
  margin_cap_ratio: 0.5          # R1: margin_abs <= 0.5 * precio_final_eur
  sold_only: true                # R2: lot_status indica vendidos (sold/closed/vendido/...)
  max_year_gap: 3               # R3: |year_bca - anio_ganvam| <= 3

demand:
  use_brand_share: true          # usa share_marca_%_{region} (normalizado 0..1)
  use_units_abs: true            # usa units_abs_{region} (normalizado 0..1)
  use_concentration_penalty: false  # opcional: penaliza dominancia de un solo modelo
  weight_brand_share: 0.5        # pesos relativos; internamente se re-normalizan
  weight_units_abs: 0.5
  weight_concentration: 0.0
```

#### Variables explicadas

- `alpha`: **0..1**. Peso del **margen** en el score. La **rotación** pesa `1 - alpha`.  
  - Ej.: `alpha=0.6` → margen 60%, rotación 40%.  
- `global_filters.margin_cap_ratio`: límite de margen relativo al precio (`0.5` = 50%).  
- `global_filters.sold_only`: si `true`, solo usa lotes vendidos (`lot_status` con “sold”, “vendido”, “sale ended/closed”).  
- `global_filters.max_year_gap`: tolerancia de discrepancia entre `year_bca` y `anio_ganvam`.  
- `demand.*`: **sensores de demanda** para dar más peso a lo que realmente se mueve:  
  - `use_brand_share`: usa la **cuota de marca** regional.  
  - `use_units_abs`: usa el **volumen absoluto** regional.  
  - `use_concentration_penalty`: si `true`, resta puntos cuando un solo modelo concentra demasiado la marca.  
  - `weight_*`: pesos relativos de cada señal; se re-normalizan a 1 si no suman 1.

> El **factor de demanda** multiplica la contribución del margen, manteniendo una “foto” estática de dónde hay más profundidad de mercado.

### Lista de consultas

Cada consulta tiene este esquema:

```yaml
- name: <texto libre para identificar la consulta>
  region: bcn | cat | esp
  filters:
    max_age_years: <int>    # opcional
    min_price: <float>      # opcional
    max_price: <float>      # opcional
  prefer_fast: true|false       # aumenta el peso de rotación
  ignore_rotation: true|false   # usa SOLO margen (anula alpha → 1.0)
  brand_only: true|false        # ranking por MARCA en lugar de modelo
  top_n: <int>                  # cuántos resultados
```

Casos especiales (agregados por país/subasta para un modelo concreto):  
```yaml
- name: Seat León 2023: país/subasta más baratos
  type: seat_leon_2023
```

## Qué devuelve

- Un CSV por consulta con columnas como:  
  `marca, modelo, anio, combustible_norm, precio_medio, margin_abs_medio, n_listings, score, mix_0_3, rank_model_region`  
- Para consultas especiales (tipo `seat_leon_2023`):  
  - `... - by_country.csv` con `precio_medio_modelo_pais` por `sale_country`.  
  - `... - by_subasta.csv` con `precio_medio_modelo_pais_subasta` por `sale_country + sale_name`.

## Buenas prácticas

- Define primero **filtros globales** → obtendrás una “foto” más limpia.  
- Ajusta `alpha` según tu sesgo:  
  - **0.6** (recomendado) → equilibrio a favor del margen.  
  - **0.5** si quieres mitad margen/mitad rotación.  
- Activa `demand` si quieres priorizar mercados con más profundidad (share/volumen).  
- Si buscas **marca** para una zona: usa `brand_only: true`.  
- Si te corre prisa vender en una zona: usa `prefer_fast: true` y rango de precio.

## Errores comunes

- Falta de columnas mínimas → revisa los nombres exactos y la presencia de `{region}` en métricas DGT.  
- Campos con `%` en 0–100 → el motor normaliza automáticamente a 0–1.  
- `lot_status` heterogéneo → se normaliza a minúsculas; si tienes otros valores de “vendido”, añádelos al runner.

## Extensiones futuras (Fase 2)

- Interfaz **Streamlit** con los mismos parámetros (YAML como preset).  
- Cartas y dashboards por región/marca/modelo.  
- Exportables y presets guardados.

---

Cualquier duda, edita `queries_examples.yaml` y vuelve a ejecutar `run_queries.py`. Si necesitas nuevos tipos de consulta, dime el formato que quieres y lo añadimos.

