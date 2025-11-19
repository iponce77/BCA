# Recomendador BCA+DGT — Fase 1.5 (consultas orientadas a negocio)

Este paquete permite definir **preguntas** sobre la base enriquecida BCA+DGT y obtener **CSV** en el **orden exacto de columnas de negocio**.

> **Orden **único** de columnas en TODAS las salidas de listados:**
>
> `link_ficha, make_clean, modelo_base, segmento, year, mileage, fuel_type, transmission-sale_country, auction_name, end_date, winning_bid, precio_final_eur, precio_venta_ganvam, margin_abs, vat_type, units_abs_bcn, units_abs_cat, units_abs_esp`

## Cambios clave (esta versión)

- **“Vehículo barato” = *vehículo óptimo*** por defecto (`selection: cheapest`), con desempate por `score` (margen × demanda + rotación).  
- **Conformador de salida**: todos los listados pasan por un **formateador** que asegura **solo** las columnas solicitadas y en ese **orden**.  
- Nuevos **tipos de consulta** para responder a *“preguntas GPT Pro”*:
  1. **`q1`** — *Mejor subasta* para un modelo concreto (devuelve **Top-N listados** + **ranking de subastas**).
  2. **`q2`** — *Precio por marca*: para una **marca**, ordena por precio el **vehículo óptimo** de cada **modelo_base**.
  3. **`q3`** — *Precio por segmento*: para un **segmento**, ordena por precio el **vehículo óptimo** (Top-N).
  4. **`q4`** — *Dónde se vende mejor este modelo/año*: ranking **BCN/CAT/ESP** ponderando `units`, `share`, `YoY`, `coef_var` y `HHI`.
  5. **`q5`** — *Mejor fuel y gap* por región para `modelo_base`+`año` usando `best_fuel_*`, `is_best_fuel_*`, `row_vs_best_fuel_%_*`.

- Compatibilidad con consultas **genéricas** (V1), ahora también normalizadas a columnas de negocio.

## Archivos

- `bca_invest_recommender.py` — motor de scoring/selección y consultas especiales (**vehículo óptimo** por defecto).  
- `run_queries.py` — runner de YAML + conformado de columnas + outputs extra (rankings).  
- `queries_examples.yaml` — plantilla con ejemplos completas.  
- Dataset ejemplo: `bca_enriched_with_ine.xlsx` / `bca_enriched_with_ine_2025-10.xlsx`.

## Datos mínimos requeridos

Mínimos: `precio_final_eur, precio_venta_ganvam, margin_abs, sale_country`  
Recomendados: `make_clean, modelo_base(_x/_y/_match), segmento, year/anio, mileage/km, fuel_type/combustible_norm, transmission, vat_type, auction_name/sale_name, end_date, winning_bid, units_abs_{bcn|cat|esp}`

> El formateador hace *coalesce* inteligente: p. ej. `modelo_base ← modelo_base_x → _y → _match → modelo`, `year ← year|anio|Año`, `mileage ← mileage|km|kilómetros|odómetro`, `fuel_type ← fuel_type|combustible_norm`, `auction_name ← sale_name` y construye `transmission-sale_country`.

## Cómo ejecutar

```bash
python run_queries.py --data "bca_enriched_with_ine.xlsx" --yaml "queries_examples.yaml" --outdir "salida"


---

## 3️⃣ `recommender/README.md` (BCA Invest Recommender)

```markdown
# BCA Invest Recommender

Este módulo implementa el recomendador de compras para BCA Invest:
dado un conjunto de lotes y sus métricas de mercado, calcula un **score**
que combina:

- Margen esperado,
- Demanda de mercado,
- Rotación rápida (mix de antigüedad),

y devuelve un ranking de “mejores vehículos para comprar”.

## Ficheros principales

- `bca_invest_recommender.py`  
  Implementa la clase `BCAInvestRecommender`, que:
  - carga el dataset enriquecido,
  - aplica filtros de inventario,
  - calcula el score,
  - devuelve las recomendaciones.

- `run_queries.py`  
  Script de entrada que:
  - lanza consultas/preparaciones sobre la tabla enriquecida,
  - inicializa `BCAInvestRecommender`,
  - guarda las recomendaciones a CSV.

---

## Inputs esperados

El recomendador espera trabajar sobre la tabla BCA enriquecida por
`bca_enrichment_pipeline.py`, con columnas como:

- Atributos BCA:
  - `make_clean`, `modelo_base_x`, `segmento`, `year_bca`, `mileage`,
  - `fuel_type`, `transmission`, `sale_country`, etc.
- Precio:
  - `precio_final_eur`, `precio_venta_ganvam`, `margin_abs`, `vat_type`.
- Mercado (por región):
  - `units_abs_bcn`, `units_abs_cat`, `units_abs_esp`,
  - `share_marca_%_esp`, `share_combustible_%_esp`, etc.
  - `mix_0_3_%_bcn`, `mix_0_3_%_cat`, `mix_0_3_%_esp` (nuevo mix corregido),
  - `YoY_%_esp`, `YoY_weighted_esp`, `Growth_3a_%_esp`, etc.

---

## Lógica de scoring

### 1. Factor de demanda

Método: `_demand_factor(region: str) -> pd.Series`

Combina distintas señales de demanda según la configuración:

- Unidades absolutas (`units_abs_{region}`).
- Cuota de marca (`share_marca_%_{region}`).
- Otros componentes configurables.

Resultado: un factor en escala [0,1] aproximadamente, que indica
“qué tanta demanda tiene este cohorte en el mercado”.

### 2. Proxy de rotación rápida

Método: `_fast_rotation_proxy(region: str) -> pd.Series`

Define una señal de **rotación** basada en la estructura de edad:

1. Busca la columna `mix_0_3_%_{region}`.
2. Si está en [0,100], la lleva a [0,1] dividiendo por 100.
3. Rellena NaN con 0.

Con el cambio del pipeline de enriquecimiento, `mix_0_3_%_{region}` ahora representa:

> *Porcentaje de transmisiones de ese modelo+combustible en la región
> que se realizan con vehículos de 0–3 años.*

Es decir, cuanto mayor es el mix 0–3, más “rápidamente se mueve” ese modelo
en el mercado → mejor proxy de rotación.

### 3. Normalización y peso margen vs rotación

Método: `_composite_score(region: str) -> pd.Series`

1. Normaliza margen absoluto:  
   `margin = normalize(margin_abs)`
2. Calcula el factor de demanda:  
   `demand = _demand_factor(region)`
3. Combina ambos:  
   `margin_demand = margin * demand`
4. Calcula rotación normalizada:  
   `rot = normalize(_fast_rotation_proxy(region))`
5. Pondera según la configuración (`alpha` y `rotation_boost`):

   ```python
   alpha = cfg.alpha_margin  # peso del margen
   rot_weight = max(0, min(1, 1 - alpha + cfg.rotation_boost))
   mar_weight = max(0, min(1, alpha))

   score = mar_weight * margin_demand + rot_weight * rot

