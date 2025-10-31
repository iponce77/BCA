# Proyecto DGT Transmisiones

## 1. Objetivo

Este proyecto procesa los **archivos de transmisiones de la DGT** para generar un análisis de mercado de los vehículos transmitidos en España y producir agregados que luego se usan para enriquecer la base BCA.

**Formato de entrada recomendado:** `.parquet` (se admiten también `.csv`, `.csv.gz` o `.zip` con CSVs).  
**Formato de salida oficial:** **`.parquet`** (`agg_transmisiones.parquet`, `agg_transmisiones_ine.parquet`, etc.).

---

## 2. Qué hace el proyecto

1. **Carga de datos DGT**  
   - Lectura de ficheros `.parquet`, `.csv`, `.csv.gz` o `.zip` con múltiples CSV.  
   - Soporte para múltiples meses (ventanas rolling de N meses o corte anual).

2. **Normalización y limpieza**  
   - Columnas resultantes: `marca`, `modelo`, `modelo_base`, `modelo_normalizado`, `make_clean`.  
   - Normalización de marcas y combustibles con diccionarios (`mappings_loader`).  
   - Limpieza de VIN, fechas, antigüedad y codificación.

3. **Normalización de INE (municipios)**  
   - Normaliza `codigo_ine` (p. ej. `28065.0` → `28065`, `4561` → `04561`).  
   - Se usa exclusivamente `codigo_ine` (no `localidad`) para agregados a nivel municipio.

4. **Agregación**  
   - **Por provincia (`agg_transmisiones.parquet`)**: `marca_normalizada`, `modelo_normalizado`, `anio`, `combustible`, `provincia`, `codigo_provincia`, `yyyymm` + métricas (`unidades`, `antiguedad_media`, percentiles, YoY, shares, etc.).  
   - **Por municipio / INE (`agg_transmisiones_ine.parquet`)**: claves `marca_normalizada`, `modelo_normalizado`, `anio`, `combustible`, `codigo_ine`, `yyyymm` + métricas similares.

5. **Exportación**  
   - Resultados oficiales en **Parquet**. (Históricamente hubo CSVs; el pipeline produce Parquet oficialmente).  
   - Generación de snapshots por periodos: 12m, 24m y YTD cuando corresponde.

---

## 3. Convenciones de nombres DGT

El pipeline infiere el `YYY YMM` (yyyymm) a partir del **nombre del fichero**. Se admiten los dos esquemas más usados:

- `resultado_mensual_trf_YYYYMM.parquet`  
- `dgt_transmisiones_YYYYMM.parquet`

Además, cualquier fichero cuyo nombre contenga una secuencia `YYYYMM` será aceptado (p. ej. `foo_202406_bar.parquet`).

La columna `transmision` se rellena con `YYYY-MM` inferido a partir del nombre del fichero cuando procede.

---

## 4. Recursos y normalizador

- El **normalizador** utilizado para enriquecer DGT está en la raíz: `normalizacionv2.py` (por compatibilidad con la automatización).  
- La **whitelist** por defecto ahora está en la raíz: `whitelist.xlsx` (si tienes otra ubicación, pásala con `--whitelist-xlsx`).  
- Los **pesos** para el normalizador (opcional) están en la raíz como `weights.json`. El proceso de automatización (`scripts/DGT/automatizacion_dgt.py`) usa `NORMALIZADOR_WEIGHTS_PATH` si está definido, y si no, intenta usar `weights.json` en la raíz como fallback.

---

## 5. Ejecución en local

### Comando básico (rolling)
```bash
python "union transmisiones/etl_transmisiones.py" --input-dir "entrada_dgt" --out-dir "salida" --mode rolling --months 12
