# Unión transmisiones (DGT → métricas de mercado)

Este módulo se encarga de transformar los datos de transmisiones de la DGT
en métricas agregadas de mercado por modelo, combustible, zona geográfica y mes.
Es la “capa de mercado” sobre la que luego se apoya el pipeline de BCA.

## Objetivo

A partir de ficheros brutos de la DGT (transmisiones de vehículos), se generan
tablas agregadas con:

- Volumen de transmisiones (`unidades`)
- Antigüedad media y percentiles (`antiguedad_media`, `p50_antiguedad`, `p75_antiguedad`)
- Mix de antigüedad en tramos (0–3, 4–7, 8+ años) `%_0_3`, `%_4_7`, `%_8_mas`
- Cuotas y shares (vía `add_shares`)
- Tendencias de unidades (`YoY_unidades_%` en la capa DGT)
- Agregaciones por provincia y por código INE (para BCN/CAT/ESP)

Estas métricas se exportan a ficheros `agg_transmisiones*.parquet/csv`, que
serán consumidos más adelante por `bca_enrichment_pipeline.py`.

---

## Estructura de ficheros

- `etl_transmisiones.py`  
  Script principal de ETL. Orquesta la carga de datos, la estandarización del esquema,
  las agregaciones mensuales y la exportación de ficheros agregados.

- `dgt_schema.py`  
  Define el esquema de los ficheros de la DGT y contiene la función
  `standardize_lazyframe`, que:
  - Normaliza tipos de columna (fechas, numéricos, strings).
  - Calcula `antiguedad_anios`.
  - Deriva `anio` (año de matriculación) cuando falta, a partir de fecha de transmisión.

- `metrics.py`  
  Aquí se definen las funciones de agregación sobre las transmisiones:
  - `aggregate_month`: agregación por provincia (`AGG_KEYS`).
  - `aggregate_month_ine`: agregación por ámbito INE (`AGG_KEYS_INE`, basado en `codigo_ine`).
  - `add_yoy`: añade `YoY_unidades_%` mensual.
  - `add_shares`: añade shares relativos por marca, combustible, etc.

- `mappings_loader.py`  
  Utilidades para cargar mapeos (por ejemplo, provincia → código INE).

- `utils_io.py`  
  Funciones de entrada/salida (lectura de múltiples ficheros, guardado a parquet/csv, etc.).

---

## Flujo de trabajo

### 1. Estándar DGT

`etl_transmisiones.py`:

1. Lee uno o varios ficheros brutos de transmisiones (CSV/parquet).
2. Aplica `dgt_schema.standardize_lazyframe`:
   - Limpieza de columnas.
   - Cálculo de `antiguedad_anios`.
   - Normalización de `marca_normalizada`, `modelo_normalizado`, `combustible`, etc.

### 2. Agregación mensual

En `metrics.aggregate_month` y `aggregate_month_ine`:

- Claves de agregación:
  - Por provincia:  
    `["marca_normalizada","modelo_normalizado","anio","combustible","provincia","codigo_provincia","yyyymm"]`
  - Por INE:  
    `["marca_normalizada","modelo_normalizado","anio","combustible","codigo_ine","yyyymm"]`

- Métricas calculadas:
  - `unidades` = número de transmisiones en el grupo.
  - `antiguedad_media`, `p25_antiguedad`, `p50_antiguedad`, `p75_antiguedad`.
  - Contadores de antigüedad por tramo:
    - `cnt_0_3`  (<= 3 años),
    - `cnt_4_7`  (3–7),
    - `cnt_8_mas` (>7).
  - Mix de antigüedad por tramo (a nivel del grupo):
    - `%_0_3`   = 100 * `cnt_0_3` / `unidades`
    - `%_4_7`   = 100 * `cnt_4_7` / `unidades`
    - `%_8_mas` = 100 * `cnt_8_mas` / `unidades`

> Estos `%_0_3/%_4_7/%_8_mas` son la materia prima que más tarde se
> re-agregará a nivel de modelo dentro de `bca_enrichment_pipeline.py` para
> construir el `mix_0_3_%_{bcn,cat,esp}` usado por el recomendador.

### 3. Tendencias y shares

Después de la agregación:

- `add_yoy`:
  - Desplaza `yyyymm` 12 meses atrás y calcula `YoY_unidades_%` como  
    `(unidades_t - unidades_{t-12})*100 / unidades_{t-12}` cuando hay base.

- `add_shares`:
  - Calcula shares de unidades por marca, modelo, combustible, etc. dentro de cada ámbito.

### 4. Exportación

`etl_transmisiones.py` ejecuta `export_for_period`, que:

- Filtra el período deseado (por ejemplo rolling 12 meses).
- Genera:
  - `agg_transmisiones.parquet/csv` (por provincia)
  - `agg_transmisiones_ine.parquet/csv` (por código INE)

Estos ficheros son los que se usan en la fase de enriquecimiento BCA.

---

## Ejemplo de ejecución

Ejemplo típico (rolling 12 meses):

```bash
python "union transmisiones/etl_transmisiones.py" \
  --mode rolling \
  --months 12 \
  --input_dir data/dgt_raw \
  --output_dir data/agg_transmisiones
