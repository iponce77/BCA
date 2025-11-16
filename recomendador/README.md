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
