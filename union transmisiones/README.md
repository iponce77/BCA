# Proyecto DGT Transmisiones

## 1. Objetivo

Este proyecto procesa los **archivos de transmisiones de la DGT** en formato CSV/ZIP (`resultado_mensual_trf_YYYYMM.csv.gz`) para generar un análisis de mercado de los vehículos transmitidos en España.

El análisis es la base para el **enriquecimiento de la base de datos BCA**: a través de los agregados que genera este pipeline (`agg_transmisiones.csv` y `agg_transmisiones_ine.csv`) se podrán asignar indicadores de atractivo o relevancia de mercado a los vehículos de BCA.

---

## 2. Qué hace el proyecto

1. **Carga de datos DGT**  
   - Lectura de ficheros `.csv`, `.csv.gz` o `.zip` con múltiples CSV.  
   - Soporte para múltiples meses (rolling windows de 12, 24 meses o corte YTD).  

2. **Normalización y limpieza**  
   - Columnas: `marca`, `modelo`, `modelo_base`, `modelo_normalizado`, `make_clean`.  
   - Normalización de combustible y marcas con diccionarios (`mappings_loader`).  
   - Limpieza de VIN, fechas, antigüedad y codificación.  

3. **Normalización de INE (municipios)**  
   - Si el código viene como `28065.0` → se transforma en `28065`.  
   - Si tiene 4 dígitos (ej. `4561`) → se completa con un `0` delante (`04561`).  
   - Si no es numérico → se descarta (`null`).  
   - Se usa **exclusivamente el campo `codigo_ine`** (no `localidad`, ya que no siempre es fiable).  

4. **Agregación**  
   - **Por provincia (`agg_transmisiones.csv`)**  
     Incluye métricas de unidades, antigüedad media, percentiles, YoY y shares (%).  
   - **Por municipio (`agg_transmisiones_ine.csv`)**  
     Incluye métricas similares pero con clave `codigo_ine`.  

5. **Exportación**  
   - Resultados en **CSV y Parquet**.  
   - Generación de snapshots por periodos: 12 meses, 24 meses, YTD.  

---

## 3. Estructura de salida

### `agg_transmisiones.csv` (nivel provincia)
- Claves: `marca_normalizada`, `modelo_normalizado`, `anio`, `combustible`, `provincia`, `codigo_provincia`, `yyyymm`.
- Métricas:  
  - `unidades`, `antiguedad_media`, `p25_antiguedad`, `p50_antiguedad`, `p75_antiguedad`  
  - `%_0_3`, `%_4_7`, `%_8_mas`  
  - `unidades_prev`, `YoY_unidades_%`, `share_prov_%`, `share_esp_%`

### `agg_transmisiones_ine.csv` (nivel municipio / INE)
- Claves: `marca_normalizada`, `modelo_normalizado`, `anio`, `combustible`, `codigo_ine`, `yyyymm`.
- Métricas:  
  - `unidades`, `antiguedad_media`, `p25_antiguedad`, `p50_antiguedad`, `p75_antiguedad`  
  - `%_0_3`, `%_4_7`, `%_8_mas`

---

## 4. Ejecución en local

### Comando básico
```powershell
python etl_transmisiones.py --input-dir "entrada_dgt" --out-dir "salida" --mode rolling --months 12
```

### Parámetros principales
- `--input-dir` → Carpeta con los CSV originales de la DGT.  
- `--out-dir` → Carpeta donde se guardan los resultados.  
- `--mode` →  
  - `rolling` → genera salidas de 12m, 24m y YTD.  
  - `single` → genera solo un corte específico.  
- `--months` → Número de meses para ventana rolling (ej. 12, 24).  

### Ejemplo extendido
```powershell
python etl_transmisiones.py --input-dir "entrada_dgt" --out-dir "salida" --mode rolling --months 24
```

Esto genera:
- `salida/agg_transmisiones.csv`  
- `salida/agg_transmisiones_ine.csv`  
- `salida/snapshots/12m/`  
- `salida/snapshots/24m/`  
- `salida/snapshots/ytd/`

---

## 5. Integración con BCA

El **punto de unión** con la base de datos de BCA será el archivo `agg_transmisiones.csv` (y en algunos casos también `agg_transmisiones_ine.csv`).

Ejemplo de uso:
- Tengo un **SEAT León gasolina de 2023 en Terrassa**.  
- Si existe el mismo segmento en `agg_transmisiones_ine.csv` (con INE de Terrassa), se asigna puntuación con granularidad municipal.  
- Si no existe a ese nivel, se puede usar `agg_transmisiones.csv` (provincia) o niveles más generales (modelo sin combustible).  
- El campo clave para enlazar es `marca_normalizada + modelo_base/modelo_normalizado + anio + combustible`.  
