# BCA – Pipeline de Datos, Pricing y Recomendador (BCA + Ganvam + DGT)

Este repositorio contiene **todo el pipeline de datos** para analizar subastas de BCA
y generar un **recomendador de inversión** basado en:

1. Datos internos de BCA (scraping mensual).
2. Tarifas de **Ganvam** (precio de venta/mercado).
3. Transmisiones de la **DGT** (demanda real, mix de antigüedad, tendencias).
4. Un **motor de recomendación** (BCA Invest Recommender) que combina margen,
   demanda y rotación para priorizar qué vehículos comprar.

---

## 🧩 Visión global del pipeline

A nivel conceptual, el sistema tiene **4 grandes bloques**:

1. **BCA – Fases 1 y 2 (mensual)**  
   Scraping de BCA Europe:
   - Fase 1A/1B → fichas completas de subasta.
   - Fase 2 → información económica/post-subasta y merge.

   Salida: un lote mensual **BCA enriquecido internamente** (vehículos, atributos, precios).

2. **GANVAM – Precio de venta**  
   Scraping y normalización de las tarifas Ganvam:
   - Fase 1 → descarga jerarquía y endpoints.
   - Fase 2 → descarga y normalización de precios por modelo/año/fuel.

   Salida: un **master Ganvam** (Parquet) con precios de referencia de mercado,
   que se cruza posteriormente con BCA.

3. **DGT – Transmisiones y mercado (INE)**  
   ETL de ficheros de transmisiones de la DGT:
   - Limpieza y estandarización del esquema.
   - Agregación por modelo, combustible, región (BCN/CAT/ESP) y periodo.
   - Cálculo de unidades, mix de antigüedad, shares y tendencias.

   Salida: agregados tipo `agg_transmisiones_ine.parquet` que se usan en el
   **enriquecimiento BCA+INE**.

4. **Recomendador BCA Invest**  
   Una vez unidas todas las piezas (BCA + Ganvam + DGT), se genera una tabla
   `bca_enriched_with_ine.*` sobre la que se ejecutan consultas de negocio y un
   **score de recomendación**:

   - Margen esperado (`margin_abs`).
   - Demanda de mercado (`units_abs_*`, shares).
   - Rotación rápida (`mix_0_3_%_{bcn,cat,esp}` corregido).
   - (Opcional) Tendencia suavizada (`YoY_weighted_{region}`).

---

## 🗺️ Fases / vías de datos

### 1. Vía BCA – Scraping y enriquecimiento interno

**Objetivo**: obtener un Excel/Parquet mensual de BCA con todos los lotes, atributos y
datos económicos post-subasta.

Scripts principales (raíz del repo):

- `FASE1ANEW.py`, `Fase1A_cloud.py`, `Fase1A_playwright_compat.py`  
  → **Fase 1A**: scraping de URLs de fichas de subasta (sin login).

- `Fase1B_cloud.py`, `Fase1B_enrich.py`  
  → **Fase 1B**: scraping detallado de cada ficha con login, enriqueciendo el Excel.

- `Fase2_cloud.py`  
  → **Fase 2**: scraping económico/post-subasta, merge inplace y reporting de errores.

- `MERGE_ONLY.py`, `add_fijos_y_precio_final.py`, `add_segmento.py`, etc.  
  → utilidades para completar columnas de precio final, tipo de IVA, segmentos, etc.

Orquestación en GitHub Actions:

- `.github/workflows/bca_fases_1a_1b.yml`  
  Ejecuta Fase 1A + 1B en cloud.

- `.github/workflows/bca_fase_2.yml`  
  Ejecuta Fase 2 (económico) en cloud.

- `.github/workflows/bca_monthly.yml` / `bca_monthly_manual.yml`  
  Orquestan el **cierre mensual** completo:
  - descargan el master Ganvam desde Drive,
  - ejecutan matching/enriquecimiento,
  - suben los outputs mensuales a Drive.

Salida típica de esta vía:

- `bca_enriched.parquet` (o equivalente) con la base BCA lista para cruzar con Ganvam y DGT.

---

### 2. Vía Ganvam – Precio de venta (carpeta `precio venta/`)

**Objetivo**: automatizar la descarga, normalización y publicación de las tarifas Ganvam
para su uso como precio de venta de referencia.

Carpeta: `precio venta/` (ver `README.txt` interno).

Componentes:

- `fase1.py`  
  Descarga la jerarquía completa de Ganvam:
  - Marca → Modelo → Combustible → Año → Endpoint de vehículo.

- `fase2.py`  
  Descarga las tarifas, normaliza y genera un Parquet consolidado:
  - `ganvam_fase2_normalizado.parquet`.

- `sonda.py`  
  Sonda semanal:
  - detecta si Ganvam ha publicado un nuevo periodo trimestral,
  - actualiza `ganvam_state.json` y dispara ejecuciones automáticas.

- `upload_ganvam_parquet.py`  
  Sube el master Ganvam a Google Drive (carpeta configurada vía secrets).

Orquestación:

- `.github/workflows/ganvam.yml`  
  - Corre la sonda todos los lunes 06:00 UTC.
  - Si hay un nuevo periodo Ganvam, ejecuta Fase 1 y 2 y actualiza el Parquet en Drive.

Salida principal:

- `ganvam_fase2_normalizado.parquet` (en Drive), que luego se usa como **master de precios**
  en los scripts de enriquecimiento (ej. `bca_enrich_all.py`, `bca_enrich_lib.py`).

---

### 3. Vía DGT – Unión transmisiones (carpeta `union transmisiones/`)

**Objetivo**: procesar archivos de transmisiones de la DGT y producir agregados de mercado
para una capa de análisis y enriquecimiento BCA + INE.

Carpeta: `union transmisiones/` (ver `README.md` interno).

Scripts clave:

- `dgt_schema.py`  
  Define el esquema de entrada (ficheros DGT) y estandariza:
  - fechas,
  - `antiguedad_anios`,
  - `marca_normalizada`, `modelo_normalizado`, `combustible`, etc.

- `metrics.py`  
  Agrega transmisiones por:
  - provincia y mes (`agg_transmisiones.parquet`),
  - código INE y mes (`agg_transmisiones_ine.parquet`),
  y calcula:
  - `unidades`,
  - edad media y percentiles,
  - mix de antigüedad por tramo (0–3, 4–7, 8+),
  - shares,
  - YoY (en capa DGT).

- `etl_transmisiones.py`  
  Orquesta el ETL:
  - lee todos los ficheros de una carpeta,
  - aplica `dgt_schema` + `metrics`,
  - genera los Parquet agregados para ventanas rolling de N meses.

- `mappings/` + `mappings_loader.py`  
  Mapeos de provincias / municipios a códigos INE y otras claves geográficas (BCN/CAT/ESP).

Orquestación:

- `.github/workflows/union_transmisiones.yml`  
  - Corre cada mes (cron) o manualmente.
  - Descarga los Parquet DGT desde Drive (`DGT_PARQUET_FOLDER_ID`).
  - Ejecuta `etl_transmisiones.py` en modo rolling (por defecto 12 meses).
  - Sube `agg_transmisiones_ine.parquet` a la carpeta mensual de BCA en Drive.

- `.github/workflows/dgt_automatizacion.yml`  
  - Automatiza la **captación y normalización** de DGT (si está configurado).

Salida principal:

- `agg_transmisiones_ine.parquet` (en Drive), con métricas agregadas por
  (`marca`, `modelo`, `anio`, `combustible`, `codigo_ine`, `yyyymm`, métricas…).

---

### 4. Enriquecimiento BCA + Ganvam + INE

Aquí confluyen las tres vías anteriores.

#### 4.1. Enriquecimiento BCA + Ganvam

Scripts (raíz):

- `bca_enrich_lib.py`  
  Librería que implementa:
  - matching BCA ↔ master Ganvam (strict → relax),
  - enriquecimiento de BCA con tarifas y métricas de ROI,
  - análisis por modelo/segmento.

- `bca_enrich_all.py`  
  CLI que orquesta:
  - carga configs (`merge_config.yaml`),
  - corre el matching,
  - corre el enriquecimiento,
  - genera:
    - `bca_enriched.xlsx`,
    - `bca_enriched_analysis.xlsx`,
    - `audit_matching.xlsx`,
    - checkpoints en Parquet.

Este bloque se usa tanto localmente como dentro del workflow mensual de BCA.

#### 4.2. Enriquecimiento BCA + INE (DGT)

Script central:

- `bca_enrichment_pipeline.py`  
  - Lee un BCA enriched (BCA + Ganvam) y `agg_transmisiones_ine.parquet`.
  - Normaliza los datos de INE (regiones, claves de modelo/combustible).
  - Calcula métricas de mercado por cohorte y modelo:
    - `units_abs`, shares, rankings,
    - **tendencias**: `YoY_%`, `Growth_3a_%`, `trend_flag`,
    - **tendencia suavizada**: `YoY_weighted` (nuevo),
    - **estructura de edad**: `antiguedad_media`, `p50_antiguedad`, `p75_antiguedad`,
    - **mix de antigüedad corregido** (`mix_0_3_%`, `mix_4_7_%`, `mix_8mas_%` agregados por modelo+fuel+región),
    - dominancia de modelo, HHI, estabilidad (stddev, coef_var).
  - Empareja estas métricas a cada fila BCA por región:
    - sufijos `_bcn`, `_cat`, `_esp`.

Entrada típica:

- `bca_enriched.parquet` (BCA+Ganvam).
- `agg_transmisiones_ine.parquet` (DGT/INE).

Salida:

- `bca_enriched_with_ine.parquet` / `.xlsx` – base final para el recomendador.

Orquestación:

- `.github/workflows/bca_enrich_with_ine.yml`  
  - Descarga `bca_enriched.parquet` y `agg_transmisiones_ine.parquet` desde Drive.
  - Ejecuta `bca_enrichment_pipeline.py`.
  - Sube `bca_enriched_with_ine.parquet` a la carpeta mensual.

---

### 5. Recomendador BCA Invest (`recomendador/`)

Carpeta: `recomendador/` (ver README específico dentro).

Componentes:

- `bca_invest_recommender.py`  
  Implementa:
  - el motor de scoring,
  - lógica de “vehículo óptimo”,
  - consultas como:
    - mejor subasta por modelo,
    - precio por marca/segmento,
    - mejor región para vender/comprar, etc.

- `run_queries.py`  
  CLI para lanzar consultas a partir de un YAML de queries:
  - carga `bca_enriched_with_ine.*`,
  - aplica filtros y scoring,
  - genera CSVs listos para negocio.

- `queries_examples.yaml`  
  Ejemplos de consultas de negocio parametrizadas.

Orquestación:

- `.github/workflows/Recomendador BCA (fase1).yml`  
  Permite ejecutar el recomendador contra un lote mensual, generando
  listados listos para consumo.

---

## ⚙️ YAMLs y configuración visualmente

Listado de los YAML relevantes:

### Workflows de GitHub Actions (`.github/workflows/`)

- `bca_fases_1a_1b.yml`  
  → Scraping BCA Fase 1A + 1B (fichas).

- `bca_fase_2.yml`  
  → Scraping BCA Fase 2 (post-subasta).

- `bca_monthly.yml`  
  → Cierre mensual completo (BCA + Ganvam + Enrich + uploads).

- `bca_monthly_manual.yml`  
  → Variante manual del cierre mensual.

- `ganvam.yml`  
  → Pipeline Ganvam (sonda + Fase1 + Fase2 + upload a Drive).

- `union_transmisiones.yml`  
  → ETL de transmisiones DGT → `agg_transmisiones_ine.parquet`.

- `dgt_automatizacion.yml`  
  → Automatización adicional del flujo DGT (si se usa).

- `bca_enrich_with_ine.yml`  
  → Enriquecer BCA con INE/DGT → `bca_enriched_with_ine.parquet`.

- `Recomendador BCA (fase1).yml`  
  → Ejecutar consultas del recomendador sobre el lote mensual.

- `Rankings_INE.yml`  
  → Generación de rankings basados en agregados INE (reporting).

- `transmisiones_test.yml`  
  → Workflow de pruebas para la pipeline de transmisiones.

### Otros YAML relevantes

- `union transmisiones/mappings/bca_mappings.yml`  
  → Mapeos específicos para cruzar claves DGT ↔ BCA/INE.

- `merge_config.yaml`  
  → Configuración del matching/enriquecimiento BCA ↔ master Ganvam (`bca_enrich_lib.py`).

- `recomendador/queries_examples.yaml`  
  → Plantilla de consultas de negocio para `run_queries.py`.

---

## 🧱 Estructura de carpetas (resumen)

- `union transmisiones/`  
  ETL DGT → agregados de transmisiones (ver README interno).

- `precio venta/`  
  Pipeline Ganvam (scraping y normalización de tarifas).

- `recomendador/`  
  Motor de recomendación BCA Invest + runner de queries.

- `.github/workflows/`  
  Todos los workflows de orquestación en GitHub Actions.

- Scripts raíz (`Fase1A*`, `Fase1B*`, `Fase2*`, `bca_enrich_all.py`, `bca_enrichment_pipeline.py`, etc.)  
  Punto de entrada para las distintas fases cuando se ejecuta en local o en CI.

---

## 🚀 Cómo usar este repo (muy resumido)

1. **Scraping mensual de BCA**  
   - Ejecutar Fase 1A/1B/2 (en local o via `bca_fases_1a_1b.yml` + `bca_fase_2.yml`).

2. **Actualizar Ganvam** (cuando hay nuevo periodo)  
   - Dejar que `ganvam.yml` y `sonda.py` lo hagan, o lanzar `precio venta/fase1.py` + `fase2.py`.

3. **Actualizar DGT**  
   - Ejecutar `union_transmisiones.yml` para refrescar `agg_transmisiones_ine.parquet`.

4. **Enriquecer BCA con Ganvam + INE**  
   - Usar `bca_enrich_all.py` + `bca_enrichment_pipeline.py`, o el workflow `bca_enrich_with_ine.yml`.

5. **Lanzar el recomendador**  
   - Desde `recomendador/run_queries.py` o vía `Recomendador BCA (fase1).yml`, usando el último `bca_enriched_with_ine.*`.

---

Este README resume la arquitectura completa (BCA + Ganvam + DGT + Recomendador).  
Para detalles finos de cada bloque, conviene leer también:

- `union transmisiones/README.md`
- `precio venta/README.txt`
- `recomendador/README.md`
- Comentarios en `bca_enrich_all.py` y `bca_enrichment_pipeline.py`.
