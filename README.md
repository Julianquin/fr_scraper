# fr_scraper — Scraper y pipeline de precios de vivienda (FincaRaíz)

Proyecto para **recolectar, limpiar y modelar** avisos de venta de inmuebles del
portal colombiano [FincaRaíz](https://www.fincaraiz.com.co). El flujo va desde el
*scraping* de los listados hasta un dashboard predictivo del precio de vivienda.

El repo está pensado como un MVP "lean": pocos módulos, cada uno con una
responsabilidad clara. La pieza más sólida y probada es el **scraper**; el resto
del pipeline (limpieza → features → modelo → dashboard) está en distintos grados
de madurez (ver [Estado del código](#estado-del-código)).

---

## ¿En qué consiste?

```
URLs de FincaRaíz ──► src/scraper.py ──► data/raw/<fecha>/*.csv   (snapshot por corrida)
                                             │
                       src/ingest.py ──► data/processed/housing_history.parquet  (toda la historia)
                                     └──► data/processed/housing_clean.parquet    (estado actual)
                                             │
                       src/changes.py ──► cambios de precio / nuevas / eliminadas
                                             │
                       modelo + dashboard (Streamlit)
```

Cada corrida del scraper guarda su **snapshot** en `data/raw/<fecha>/`, con las
columnas `id_inmueble` (el ID del aviso, tomado de la URL) y `fecha_recoleccion`.
Así la historia se acumula y se pueden derivar cambios de precio, altas y bajas.

1. **Scraping** — Selenium abre cada página de listado, extrae las tarjetas de
   inmuebles y luego descarga en paralelo (con `requests`) el detalle de cada
   aviso: precio, tipología, ubicación, descripción y coordenadas (lat/lon).
2. **Ingesta / limpieza** — se concatenan todos los CSV, se normalizan precios,
   se extraen habitaciones/baños/área/tipo de propiedad desde texto libre y se
   deja un único Parquet limpio.
3. **Modelado / dashboard** — un `RandomForestRegressor` estima el precio a
   partir de área, habitaciones, baños, tipo, ciudad, etc., expuesto en una app
   de Streamlit.

---

## Estructura del repositorio

```
fr_scraper/
├── src/
│   ├── config.py          # Rutas base del proyecto (data/raw, data/processed, models)
│   ├── scraper.py         # ⭐ Scraper de FincaRaíz (CLI). Genera data/raw/*.csv
│   ├── ingest.py          # Consolida los snapshots → housing_history + housing_clean (parquet)
│   ├── preprocessing.py   # Lógica de limpieza (usada por ingest.py)
│   ├── changes.py         # Deriva cambios de precio / nuevas / eliminadas desde la historia
│   ├── features.py        # Regenera urls_fincaraiz.txt (catálogo de ciudades/tipos)
│   ├── train.py           # Entrena el modelo → models/model.pkl
│   ├── app.py             # Dashboard Streamlit que carga models/model.pkl
│   └── OLD/               # Versiones antiguas del scraper
├── streamlit_dashboard.py # Dashboard alternativo, autocontenido (entrena al vuelo)
├── urls_fincaraiz.txt     # ~570 URLs de listados (una por línea) que consume el scraper
├── data/
│   ├── raw/               # Snapshots del scraper: data/raw/<fecha>/*.csv
│   ├── interim/           # (vacío)
│   └── processed/         # Parquets de ingest.py (history + clean)
├── models/                # Aquí se guardaría model.pkl (vacío por defecto)
├── notebooks/             # Notebooks de exploración / prototipos
├── Guia.md                # Ideas de producto y roadmap del MVP
└── README.md
```

---

## Requisitos e instalación

- **Python 3.12** (probado). Python 3.10+ debería funcionar.
- **Google Chrome / Chromium** instalado en el sistema. El scraper usa
  `webdriver-manager`, que **descarga automáticamente** el ChromeDriver
  compatible, así que no necesitas instalarlo a mano.

Instala las dependencias (idealmente en un entorno virtual):

```bash
cd /home/julianquin/proyectos/fr_scraper

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Uso rápido (TL;DR)

Ejecuta todo **desde la raíz del repo** (los módulos usan imports tipo `src.*`):

```bash
# 1) Scrapear (headless, 30 páginas máx. por URL)
python3 -m src.scraper --headless --max-pages 30

# 2) Limpiar y consolidar (genera housing_history + housing_clean)
python3 -m src.ingest

# 2b) Ver cambios de precio / nuevas / eliminadas (necesita ≥2 corridas)
python3 -m src.changes

# 3) Entrenar el modelo → models/model.pkl
python3 -m src.train

# 4a) Dashboard basado en el modelo entrenado
streamlit run src/app.py

# 4b) …o el dashboard autocontenido (entrena al vuelo)
streamlit run streamlit_dashboard.py
```

---

## Cómo scrapear de nuevo (guía detallada)

El scraper es `src/scraper.py` y se ejecuta como módulo:

```bash
python3 -m src.scraper [opciones]
```

### Qué hace, paso a paso

1. Lee las URLs de `urls_fincaraiz.txt` (una por línea).
2. Crea una carpeta de snapshot para la corrida: `data/raw/<fecha>/`
   (ej. `data/raw/2026-08-02/`). Así cada corrida queda guardada aparte y la
   historia se acumula sin pisarse.
3. Para cada URL calcula un **nombre de archivo** a partir de los últimos tramos
   de la URL. Ej.: `.../venta/apartaestudios/bogota/bogota-dc`
   → `data/raw/2026-08-02/venta_apartaestudios_bogota.csv`.
4. **Si ese CSV ya existe en la carpeta de hoy, lo omite** (re-ejecutar el mismo
   día es idempotente). Para forzar re-bajarlo usa `--overwrite`.
5. Recorre las páginas del listado (`/pagina2`, `/pagina3`, …) hasta
   `--max-pages` o hasta que una página venga vacía.
6. Por cada página abre el listado con Selenium y luego descarga los **detalles
   en paralelo** con hilos (`--workers`). Si el modo rápido falla, hace
   *fallback* con Selenium.
7. Escribe un CSV por URL, añadiendo las columnas **`id_inmueble`**
   (ID del aviso, extraído de la URL) y **`fecha_recoleccion`**.

### Opciones del CLI

| Opción         | Por defecto           | Descripción                                                        |
| -------------- | --------------------- | ------------------------------------------------------------------ |
| `--url-file`   | `urls_fincaraiz.txt`  | Archivo con las URLs a scrapear (una por línea).                   |
| `--out-dir`    | `data/raw`            | Carpeta destino de los CSV.                                       |
| `--max-pages`  | `50`                  | Máximo de páginas de listado a recorrer por URL.                  |
| `--delay`      | `0.0`                 | Segundos de espera entre páginas (para ir más "suave").          |
| `--headless`   | *(desactivado)*       | Corre Chrome sin ventana. Recomendado en servidores/WSL.         |
| `--overwrite`  | *(desactivado)*       | Vuelve a scrapear aunque el CSV ya exista.                        |
| `--workers`    | `8`                   | Hilos para descargar detalles en paralelo.                       |

### Escenarios típicos

**Scrapear solo lo que falta** (rápido, no re-baja lo ya descargado):

```bash
python3 -m src.scraper --headless --max-pages 30
```

**Re-scrapear TODO desde cero** (refrescar datos existentes):

```bash
python3 -m src.scraper --headless --overwrite --max-pages 30
```

**Scrapear un subconjunto propio de URLs** (crea tu propio archivo):

```bash
# mis_urls.txt con las URLs que te interesen
python3 -m src.scraper --headless --url-file mis_urls.txt --overwrite
```

**Ser más amable con el servidor** (menos hilos, con pausa entre páginas):

```bash
python3 -m src.scraper --headless --workers 4 --delay 1.5 --max-pages 20
```

> **Nota:** cada corrida escribe en `data/raw/<fecha>/`. Ejecutar dos veces el
> mismo día no duplica (omite los CSV ya escritos ese día salvo `--overwrite`);
> ejecutar en días distintos crea snapshots nuevos que alimentan la historia.
> Los ~465 CSV sueltos que ya están en `data/raw/` se tratan como el **lote
> inicial** (sin fecha) y el `ingest` los sigue leyendo.

### Columnas que genera el scraper

Cada CSV en `data/raw/` incluye, entre otras: `Título`, `URL detalle`,
`URL imagen`, `Etiquetas`, `Precio listado`, `Tipología listado`,
`Descripción breve`, `Ubicación listado`, `Publicante`, `Descripción completa`,
`Estado`, `Estrato`, `Parqueaderos`, `Latitud`, `Longitud`, etc. (el conjunto de
columnas de detalle puede variar según el aviso).

---

## Ingesta y limpieza

```bash
python3 -m src.ingest
```

- Lee **todos** los CSV de `data/raw/` de forma recursiva (todas las carpetas
  `<fecha>/` más el lote inicial suelto).
- Aplica `preprocesar_datos_finca_raiz` (`src/preprocessing.py`):
  - convierte `Precio listado` (texto con `$` y puntos) a número;
  - extrae `Habitaciones`, `Baños`, `Area_m2` y `Tipo_propiedad` desde texto;
  - normaliza `Ciudad`/`Departamento` y extrae `Barrio`;
  - crea columnas dummy de etiquetas (`Proyecto`, `Destacado`, `Nuevo`, `Oportunidad`);
  - conserva `id_inmueble` y `fecha_recoleccion` (los deriva si faltan);
  - elimina filas sin `Area_m2` o `Precio` y duplicados exactos por `(id, fecha)`.
- Escribe **dos** Parquets en `data/processed/`:
  - **`housing_history.parquet`** — toda la historia (una fila por inmueble y fecha).
  - **`housing_clean.parquet`** — estado actual (el snapshot más reciente de cada
    `id_inmueble`). Es el que consumen `train.py` y los dashboards.

---

## Rastreo de cambios (historia)

Con al menos **dos corridas del scraper en días distintos**, el módulo
`src/changes.py` deriva la evolución del mercado a partir de
`housing_history.parquet`:

```bash
python3 -m src.changes          # imprime un resumen en consola
python3 -m src.changes --save   # además guarda 3 CSV en data/processed/
```

- **📈 Cambios de precio** (`cambios_precio.csv`) — inmuebles cuyo `Precio` cambió
  entre fechas, con `precio_anterior`, `variacion` y `variacion_pct`.
- **🆕 Nuevas** (`nuevas.csv`) — `id_inmueble` cuya primera aparición es posterior
  a la primera corrida.
- **❌ Eliminadas** (`eliminadas.csv`) — inmuebles que **no** aparecen en la corrida
  más reciente (con su `ultima_vez_vista`).

> ⚠️ Que un inmueble no salga en la última corrida no prueba que se haya vendido:
> pudo cambiar de página/ranking en los resultados de búsqueda. Para confirmar una
> baja real, lo fiable es re-pedir su `URL detalle` y ver si responde 404 / "no
> disponible". La heurística actual ("no visto en la última fecha") es suficiente
> para un MVP, pero puede dar falsos positivos.

---

## Modelo y dashboard

Hay **dos** dashboards de Streamlit en el repo:

### `streamlit_dashboard.py` (raíz, autocontenido)

Carga los CSV de `data/raw/`, entrena un `RandomForestRegressor` al vuelo y
muestra la predicción interactiva. No requiere pasos previos más allá de tener
CSV scrapeados.

```bash
streamlit run streamlit_dashboard.py
```

### `src/app.py` (usa el modelo entrenado)

Entrena el modelo y lánzalo:

```bash
python3 -m src.ingest    # genera data/processed/housing_clean.parquet
python3 -m src.train     # entrena y guarda models/model.pkl (imprime el MAE)
streamlit run src/app.py # sliders para estimar el precio
```

---

## Automatización semanal (GitHub Actions)

El workflow [.github/workflows/scraper.yml](.github/workflows/scraper.yml) corre el
scraper **automáticamente cada lunes** (y a demanda desde la pestaña *Actions*):

1. Levanta un runner Ubuntu con Chrome, instala dependencias.
2. Scrapea un subconjunto acotado (`urls_demo.txt`, ciudades principales) para
   caber en el límite de tiempo del runner, y corre `ingest`.
3. **Commitea los snapshots de vuelta al repo** (`data/raw/<fecha>/` + parquets)
   para que la historia se acumule semana a semana, y los sube como *artifact*.

Ejecución manual: *Actions → Scraper semanal FincaRaíz → Run workflow* (puedes
cambiar `max_pages`, `url_file` o desactivar el commit).

Consideraciones:
- La lista completa (`urls_fincaraiz.txt`, ~570 URLs) **no cabe** en una sola
  corrida (límite de 6 h por job); por eso el default es `urls_demo.txt`. Para la
  lista completa, divídela en varios jobs o usa un runner propio.
- Como `data/` está en `.gitignore`, el workflow usa `git add -f`; los datos
  versionados crecerán con el tiempo — para producción conviene almacenamiento
  externo (S3/R2) o una BD en vez del repo.
- ⚠️ El scraping puede violar los ToS de FincaRaíz y las políticas de uso de
  GitHub Actions. Mantén el volumen bajo; para clientes, migra a datos propios.

## Estado del código

Resumen del estado actual de cada módulo (todo el pipeline funciona de punta a
punta):

| Módulo                    | Estado | Detalle                                                                                     |
| ------------------------- | :----: | ------------------------------------------------------------------------------------------- |
| `src/scraper.py`          |   ✅   | Scraper de FincaRaíz. Es la pieza principal del repo.                                        |
| `src/ingest.py`           |   ✅   | Consolida los snapshots → `housing_history.parquet` + `housing_clean.parquet`.               |
| `src/preprocessing.py`    |   ✅   | Lógica de limpieza (la invoca `ingest.py`); conserva `id_inmueble` y `fecha_recoleccion`.    |
| `src/changes.py`          |   ✅   | Deriva cambios de precio / nuevas / eliminadas desde la historia.                            |
| `src/train.py`            |   ✅   | Entrena el `RandomForestRegressor` y guarda `models/model.pkl` (imprime el MAE).            |
| `src/app.py`              |   ✅   | Dashboard que carga `models/model.pkl` (ejecuta `train` primero).                           |
| `streamlit_dashboard.py`  |   ✅   | Dashboard autocontenido; lee `data/raw/` y entrena al vuelo.                                 |
| `src/features.py`         |   ✅   | `python3 -m src.features` regenera `urls_fincaraiz.txt` a partir del catálogo de ciudades.   |

> **Nota histórica:** versiones anteriores de `train.py`, `features.py` y
> `streamlit_dashboard.py` tenían rutas/imports rotos (`DATA_INT`,
> `housing_preprocessed.parquet`, carpeta `datos/`). Ya están corregidos.

---

## Notas y buenas prácticas de scraping

- **Respeta el sitio**: usa `--delay` y un número razonable de `--workers` para
  no saturar FincaRaíz. El scraping puede violar los Términos de Servicio del
  portal; úsalo con fines educativos/de investigación y bajo tu responsabilidad.
- El sitio puede **cambiar su HTML** en cualquier momento; si dejan de salir
  datos, revisa los selectores CSS en `src/scraper.py` (`div.listingCard`,
  `span.price`, `div.project-info`, etc.).
- Ejecuta siempre **desde la raíz** del repo para que funcionen los imports
  `src.*`. En WSL/servidores usa `--headless`.

---

## Ideas de producto

En [Guia.md](Guia.md) hay 6 ideas de negocio/analítica sobre estos datos
(dashboard predictivo, informes de mercado, alertas de oportunidades, análisis
espacial, valoración para crédito hipotecario y ranking inmobiliario) junto con
un roadmap de MVP.
