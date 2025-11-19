
## 2️⃣ `bca_enrichment_pipeline_README.md`

```markdown
# Pipeline de enriquecimiento BCA + mercado (INE/DGT)

Este script (`bca_enrichment_pipeline.py`) une la información interna de BCA
(lotes, precios, atributos de vehículo) con las métricas de mercado agregadas
(provenientes de la DGT vía INE), para producir una tabla enriquecida que
sirve como base del recomendador y de análisis de pricing.

## Objetivo

A partir de:
- una tabla base de BCA (vehículos subastados, atributos, precios, etc.) y
- una tabla agregada de transmisiones (`agg_transmisiones_ine`),

el pipeline genera una tabla enriquecida con:

- Volumen de mercado (`units_abs_*`)
- Shares por marca, combustible, cohorte, año…
- Tendencias (`YoY_%`, `YoY_weighted`, `Growth_3a_%`, `trend_flag`)
- Estructura de edad y mix de antigüedad (`antiguedad_media`, `mix_0_3_%`, etc.)
- Indicadores de concentración (`HHI_marca`)
- Dominancia de modelo dentro de marca (`dominancia_modelo_marca_%`)
- Rankings internos (best fuel, rank_year_model, etc.)
- Métricas finales `*_bcn`, `*_cat`, `*_esp` que se anexan a cada vehículo BCA.

---

## Flujo de alto nivel

1. **Carga y normalización de datos BCA**
2. **Lectura y normalización de INE / agg_transmisiones**
3. **Cálculo de shares, rankings, tendencias, estabilidad, mix y dominancia**
4. **Emparejamiento por región (BCN/CAT/ESP)**
5. **Generación de la tabla final enriquecida y diccionario de columnas**

---

## Puntos clave del código

### 1. Normalización INE: `normalize_ine`

- Lee un fichero de agregados de transmisiones (INE).
- Detecta dinámicamente qué columnas corresponden a:
  - unidades (`unidades` / `units`),
  - edad (`antiguedad_media`, pXX),
  - mix (`%_0_3`, `%_4_7`, `%_8_mas`, `mix_0_3_%`, etc.).
- Construye un DataFrame `ine_norm` con un esquema homogéneo:
  - `region` (BCN/CAT/ESP u otras regiones),
  - `marca`, `modelo`, `anio`, `combustible`,
  - `unidades`, `YoY_%` (si viene del ETL previo), mix, etc.

### 2. `compute_shares_and_ranks(ine_norm)`

Función central que añade la mayoría de métricas de mercado. Internamente:

1. **`annual`**  
   - Agrega `unidades` por (`region`, `marca`, `modelo`, `anio`, `combustible`).
   - Sirve como base para shares, rankings y tendencias.

2. **Shares y rankings**  
   - Calcula:
     - `share_marca_%`, `share_combustible_%`, `share_año_%`, `share_cohorte_%`.
     - `rank_general`, `rank_brand_year`, `rank_year_fuel_model`.
     - Flags como `is_top3_brand_year`.

3. **Tendencias: `compute_trends(annual)`**  
   - Calcula:
     - `YoY_%` = (units_t − units_{t-1}) / units_{t-1} por cohorte
     - `Growth_3a_%` = (units_t − units_{t-3}) / units_{t-3}
     - `trend_flag` (1 si YoY > 10%, 0 si YoY < −10%, NaN si estable)
     - `year_rank_in_model` = ranking de años por volumen.
   - **Nuevo**: `YoY_weighted`
     - Definición:
       - `YoY_weighted = YoY_% * min(1, units_prev / 20)`
       - Reduce el impacto de YoY cuando el año anterior tiene muy pocas unidades.
     - Se usa como métrica de tendencia “suavizada” para análisis y posibles usos futuros
       en el recomendador.

4. **Estructura de edad y mix**

   - Base `estructura`: una fila por cohorte
     (`region`, `marca`, `modelo`, `anio`, `combustible`).

   - Agrega estadísticos de edad (`antiguedad_media`, `p50_antiguedad`, `p75_antiguedad`)
     promediando en `ine_norm` por las mismas claves.

   - **Mix de antigüedad (corregido)**:

     Originalmente, el mix venía de `agg_transmisiones_ine` a nivel de cohorte año-mes
     (`anio` + `yyyymm`), lo que generaba patrones 0/100.

     Ahora, se re-calcula el mix **a nivel de modelo**:

     1. A partir de `ine_norm`, se toma:
        - `region`, `marca`, `modelo`, `combustible`, `unidades`, `mix_0_3_%`, `mix_4_7_%`, `mix_8mas_%`.

     2. Se convierten esos `%` en unidades ponderadas:
        - `w_0_3  = unidades * mix_0_3_% / 100`
        - `w_4_7  = unidades * mix_4_7_% / 100`
        - `w_8mas = unidades * mix_8mas_% / 100`

     3. Se agrupa por (`region`, `marca`, `modelo`, `combustible`) y se suman:
        - `units_total`, `n_0_3`, `n_4_7`, `n_8mas`.

     4. Se reconstruyen los mix globales del modelo:
        - `mix_0_3_%  = 100 * n_0_3 / units_total`
        - `mix_4_7_%  = 100 * n_4_7 / units_total`
        - `mix_8mas_% = 100 * n_8mas / units_total`

     5. Este mix global se hace *broadcast* a todos los años (`anio`) del modelo
        en `estructura`.

     Resultado: `mix_0_3_%`, `mix_4_7_%`, `mix_8mas_%` representan ahora:

     > *“Qué porcentaje de las transmisiones de este modelo+combustible en esta región
     se realiza con vehículos de 0–3 / 4–7 / 8+ años (sobre el total del modelo)”*.

     Esta definición es la que alimenta la métrica de rotación del recomendador.

5. **Dominancia y HHI**

   - `dominancia_modelo_marca_%`: peso del modelo dentro de la marca.
   - `HHI_marca`: índice de Herfindahl de concentración de modelos por marca.

6. **Salida `features_fuel`**

   - DataFrame con una fila por
     (`region`, `marca`, `modelo`, `anio`, `combustible`) y todas las métricas de mercado
     calculadas:
     - `units_abs`, shares, rankings, `YoY_%`, `YoY_weighted`, `Growth_3a_%`, `trend_flag`,
       `year_rank_in_model`, `antiguedad_media`, mix, dominancia, HHI, estabilidad, etc.

### 3. Emparejamiento por región: `_match_region`

- Toma la tabla BCA (con `_bca_idx`) y une las métricas de `features_fuel`
  para cada región:

  - `units_abs_bcn`, `share_marca_%_esp`, `mix_0_3_%_cat`, etc.
  - `YoY_%_bcn`, `YoY_weighted_bcn`, …, `YoY_%_esp`, `YoY_weighted_esp`.

- La salida final es una tabla BCA enriquecida lista para:
  - análisis de pricing,
  - entrenamiento de modelos,
  - alimentación del recomendador.

---

## Output

La salida típica de este pipeline es un fichero tipo:

- `bca_enriched_with_ine.parquet/csv`

con una fila por vehículo BCA y columnas adicionales de mercado para BCN/CAT/ESP.

---

## Ejecutar el pipeline

Ejemplo simple (adaptar rutas / parámetros):

```bash
python bca_enrichment_pipeline.py \
  --bca_file data/bca_base.parquet \
  --ine_file data/agg_transmisiones_ine.parquet \
  --output_file data/bca_enriched_with_ine.parquet
