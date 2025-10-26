# BCA – Scraping Pipeline Cloud

Automatización de scraping para subastas de BCA Europe, **lista para ejecución en cloud con GitHub Actions**.

---

## 🔗 Fases del pipeline

- **Fase 1A**: Extrae todas las URLs de fichas de subasta activas (sin login)
- **Fase 1B**: Scraping de cada ficha, enriqueciendo el Excel (con login automático)
- **Fase 2**: Scraping económico/post-subasta, merge inplace y reporting de errores (con login automático)

**Todos los scripts generan logs profesionales, reporting por fila y CSV de errores para auditoría.**

---

## 🚀 **Ejecución en local**

1. **Clona el repo:**

    ```bash
    git clone https://github.com/tuorg/BCA.git
    cd BCA
    ```

2. **Instala dependencias:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # (o venv\Scripts\activate en Windows)
    pip install -r requirements.txt
    python -m playwright install chromium
    ```

3. **Prepara tu `.env`:**

    - Crea tu `.env` a partir de `.env.example`
    - Añade tus credenciales BCA:

        ```env
        BCA_USER=tu_usuario
        BCA_PASS=tu_password
        ```

4. **Ejecuta las fases:**

    ```bash
    python scripts/Fase1A_cloud.py
    python scripts/Fase1B_cloud.py -i fichas_vehiculos_YYYYMMDD.xlsx -s bca_storage_phase1.json
    python scripts/Fase2_cloud.py -i fichas_vehiculos_YYYYMMDD.xlsx -s bca_storage_phase1.json
    ```

    *(Ajusta la fecha al día correspondiente.)*

---

## 🤖 **Ejecución automática en cloud (GitHub Actions)**

- Workflow en `.github/workflows/bca_pipeline.yml` automatiza el pipeline diario o bajo demanda.
- Resultados, logs y CSVs de error quedan disponibles como artefactos descargables tras cada ejecución.

### **Para activar el pipeline:**

1. Sube todos los scripts y este README.
2. Configura los **secrets** en GitHub:
    - `BCA_USER`
    - `BCA_PASS`
3. Ajusta horario/trigger si lo necesitas.
4. Descarga outputs y logs desde la pestaña "Actions" → job → "Artifacts".

---

## 📄 **Outputs y reporting**

- **Excel final:** con columnas `error_fase1a`, `error_fase1b`, `error_fase2` según fase.
- **CSV de errores:** solo filas fallidas para revisión rápida.
- **Logs (`.log`):** seguimiento completo, descargables tras cada ejecución.
- **Todo queda archivado como artefacto en cada run del pipeline.**

---

## 🔒 **Seguridad**

- **No subas tu `.env` real** al repo (usa solo `.env.example`).
- Los secretos BCA deben configurarse como **Secrets** de GitHub Actions.
- Los logs y Excels pueden contener datos sensibles: descárgalos solo en dispositivos seguros.

---

## 👨‍💻 **Soporte y personalización**

- Si quieres añadir reporting (email, Slack, etc.), revisa el workflow y scripts.
- Para dudas, abre un Issue en el repo o contacta con el responsable técnico.

---

## 🛡️ **Licencia**

(Añade aquí la licencia que corresponda a tu empresa o a tu uso personal.)

---

**¿Necesitas personalización o ayuda? Abre un Issue o contacta.**
