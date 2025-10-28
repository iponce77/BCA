Fase Ganvam – Precio de Venta

Este directorio contiene la implementación de las fases 1 y 2 del pipeline de Ganvam, junto con los scripts auxiliares necesarios para su ejecución en local y en la nube (GitHub Actions + Google Drive).

---

Objetivo
Automatizar la obtención, normalización y publicación de las tarifas trimestrales de precios de venta de vehículos usadas por Ganvam, generando un dataset actualizado en formato Parquet que se sube a Google Drive para su posterior consumo por las fases posteriores del proyecto BCA.

---

Componentes principales

1. fase1.py
- Descarga la jerarquía de Ganvam (esqueleto):
  - Marca → Modelos → Combustibles → Años → Endpoints de vehículos.
- Genera ganvam_esqueleto.json con toda la estructura vigente para el periodo detectado.
- Implementa concurrencia en llamadas para mejorar el rendimiento.
- Detecta automáticamente el idPeriodo vigente si no se pasa por variable de entorno (GANVAM_PERIODO).

2. fase2.py
- Consume el ganvam_esqueleto.json de Fase 1.
- Descarga todos los precios de venta para las combinaciones marca/modelo/combustible/año.
- Genera:
  - ganvam_fase2_resultados.csv → datos completos descargados.
  - ganvam_fase2_fallos.csv → endpoints que no devolvieron datos.
- Llama a normalización con whitelist para producir el dataset final:
  - ganvam_fase2_normalizado.parquet.

3. sonda.py
- Comprueba cada semana si Ganvam ha publicado un nuevo periodo trimestral.
- Si detecta cambio, actualiza ganvam_state.json y dispara la ejecución automática de las fases 1 y 2.

4. upload_ganvam_parquet.py
- Sube el fichero ganvam_fase2_normalizado.parquet a la carpeta de Google Drive indicada en el secreto GDRIVE_FOLDER_ID.
- Reutiliza la lógica de autenticación centralizada en gdrive_auth.py (ubicado en la raíz del repo).

5. ganvam.yml (workflow de GitHub Actions)
- Planifica la ejecución:
  - Todos los lunes a las 06:00 UTC (cron).
  - O manualmente mediante workflow_dispatch.
- Fases del workflow:
  1. Sonda → detecta nuevo periodo.
  2. Pipeline → corre Fase 1 y Fase 2, normaliza, sube a Drive.

---

Variables de entorno

- GANVAM_TOKEN: JWT de autenticación Ganvam (obligatorio).
- GANVAM_EMPRESA: idEmpresa (default: 60).
- GANVAM_PERIODO: idPeriodo a procesar (se autodetecta si = 0).
- GANVAM_SLEEP: retardo entre peticiones (default: 0.35s).
- GANVAM_RETRIES: número de reintentos (default: 3).
- GDRIVE_FOLDER_ID: carpeta de destino en Google Drive.

---

Flujo de ejecución

1. sonda.py detecta nuevo periodo trimestral.
2. Ejecuta:
   - fase1.py → genera esqueleto actualizado.
   - fase2.py → descarga precios, normaliza con whitelist.xlsx.
3. Produce ganvam_fase2_normalizado.parquet.
4. upload_ganvam_parquet.py sube el archivo a Google Drive.
5. ganvam_state.json se actualiza en el repo con el último periodo procesado.

---

Notas importantes
- whitelist.xlsx y normalizacionv2.py están en la raíz del repo, no en esta carpeta.  
  fase2.py apunta correctamente usando rutas relativas.
- Si Ganvam cambia el endpoint o el formato de datos, habrá que ajustar los helpers de Fase 1 y Fase 2.
- El pipeline está diseñado para sobrescribir siempre el resultado en Drive con el mismo nombre, garantizando compatibilidad con las fases posteriores.

---

Checklist tras cada trimestre
1. Confirmar que la sonda detecta el nuevo periodo.
2. Revisar logs de ejecución en GitHub Actions:
   - JSON jerárquico generado en Fase 1.
   - Filas OK y ganvam_fase2_normalizado.parquet generado en Fase 2.
3. Verificar que el archivo se ha subido a la carpeta de Google Drive.
4. Comprobar diferencias entre whitelist y esqueleto (vehículos nuevos u obsoletos).
