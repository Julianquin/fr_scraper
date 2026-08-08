# fr_scraper — Inteligencia de mercado inmobiliario (FincaRaíz)

Pipeline para **recolectar, limpiar y analizar** avisos de venta de inmuebles del
portal colombiano [FincaRaíz](https://www.fincaraiz.com.co): scraping
**incremental**, un **store maestro** por inmueble, dashboards comerciales
(Streamlit) y **mapas interactivos** publicables en GitHub Pages. 100% descriptivo
(hay además un módulo de modelado opcional).

---

## Arquitectura

```
urls/<depto>.txt ──► src/scraper.py --incremental ──► data/raw/<fecha>/*.csv
      │                    (orden por recientes + corta al llegar a lo conocido)
      │                                   │
      │              src/ingest_master.py │ (upsert por id_inmueble)
      │                                   ▼
      │                     data/master/listings.parquet   ◄── store maestro (una fila/inmueble)
      │                                   │
      │           src/build_app_dataset.py│
      │                                   ▼
      │                     data/app/housing_clean.parquet  ◄── dataset curado del dashboard
      │                          │                    │
      │      streamlit_app.py ◄──┘                    └──► src/viz_map.py ──► docs/*.html (GitHub Pages)
      │      (precio por zona + comparador)
      │
      └── (opcional) src/ingest.py ──► housing_history + housing_clean ──► src/changes.py, src/train.py
```

**Llave de todo:** `id_inmueble` = último segmento de la URL del aviso. Con él, el
scraper sabe qué ya conoce (para cortar la paginación) y el master hace *upsert*
sin duplicar.

---

## Estructura del repositorio

```
fr_scraper/
├── src/
│   ├── config.py            # Rutas base (data/raw, data/processed, models)
│   ├── scraper.py           # ⭐ Scraper (CLI). --incremental usa el master
│   ├── master.py            # Store maestro por id_inmueble (upsert, first/last seen)
│   ├── ingest_master.py     # Upsert de los CSV al master (deriva id para CSV viejos)
│   ├── build_app_dataset.py # Dataset curado del dashboard desde el master
│   ├── preprocessing.py     # Lógica de limpieza (precio, área, ciudad, barrio…)
│   ├── ingest.py            # (alt) Consolida CSV → housing_history + housing_clean
│   ├── changes.py           # Cambios de precio / nuevas / eliminadas
│   ├── viz_map.py           # Mapas interactivos por ciudad (Folium)
│   ├── features.py          # Regenera urls_fincaraiz.txt (catálogo ciudades/tipos)
│   ├── train.py / app.py    # Modelo opcional (RandomForest) + dashboard del modelo
│   └── OLD/                 # Versiones antiguas
├── streamlit_app.py         # ⭐ Dashboard comercial (Streamlit Cloud)
├── streamlit_dashboard.py   # Dashboard autocontenido (entrena al vuelo)
├── urls/                    # URLs divididas por departamento (antioquia.txt, …)
├── urls_fincaraiz.txt       # Lista completa (~570 URLs)  ·  urls_demo.txt (subconjunto)
├── docs/                    # Sitio de GitHub Pages (index.html + mapas)
├── data/
│   ├── raw/<fecha>/         # Snapshots del scraper (gitignored)
│   ├── master/listings.parquet   # Store maestro incremental
│   ├── app/housing_clean.parquet # Dataset del dashboard (lo lee Streamlit Cloud)
│   └── processed/           # Parquets de ingest.py (history + clean)
├── .github/workflows/scraper.yml  # CI: matriz por departamento + merge + Pages
└── requirements.txt
```

---

## Requisitos e instalación

- **Python 3.12** (probado; 3.10+ debería servir).
- **Google Chrome** instalado. El scraper usa `webdriver-manager`, que descarga el
  ChromeDriver compatible automáticamente. En Linux/WSL, instala Chrome con el
  `.deb` oficial y corre siempre con `--headless`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Ejecuta todo **desde la raíz del repo** (los módulos usan imports `src.*`).

---

## Uso rápido (TL;DR)

```bash
# 1) Scrapear de forma incremental (lee el master, corta al llegar a lo conocido)
python3 -m src.scraper --headless --incremental --max-pages 50

# 2) Upsert al master + construir el dataset del dashboard
python3 -m src.ingest_master
python3 -m src.build_app_dataset

# 3) Dashboard comercial (local)
streamlit run streamlit_app.py

# 4) Mapa interactivo de una ciudad
python3 -m src.viz_map --ciudad Cali        # → reports/cali_map.html
```

---

## Scraper (`src/scraper.py`)

```bash
python3 -m src.scraper [opciones]
```

Recorre las URLs del `--url-file`, pagina cada listado (orden por **más
recientes**), y por cada aviso descarga el detalle en paralelo. Escribe un CSV por
URL en `data/raw/<fecha>/`, con `id_inmueble` y `fecha_recoleccion`.

### Opciones

| Opción            | Default              | Descripción                                                     |
| ----------------- | -------------------- | --------------------------------------------------------------- |
| `--url-file`      | `urls_fincaraiz.txt` | Archivo con URLs (una por línea).                               |
| `--max-pages`     | `50`                 | Máximo de páginas por URL.                                      |
| `--incremental`   | *(off)*              | Lee el master y **corta** al ver 2 páginas ya conocidas.        |
| `--headless`      | *(off)*              | Chrome sin ventana (obligatorio en WSL/servidores).            |
| `--workers`       | `8`                  | Hilos para descargar detalles en paralelo.                     |
| `--delay`         | `0.0`                | Segundos de espera entre páginas.                              |
| `--recycle-every` | `25`                 | Reinicia Chrome cada N URLs (evita fugas de memoria en WSL).   |
| `--overwrite`     | *(off)*              | Re-scrapea aunque el CSV del día ya exista.                    |

---

## Modo incremental (el que ahorra horas)

FincaRaíz permite ordenar por recientes (`?ordenListado=3`), así que los avisos
nuevos quedan **arriba**. El modo incremental aprovecha eso:

1. Carga los precios conocidos del **master** (`data/master/listings.parquet`).
2. Scrapea con orden por recientes; en cuanto **2 páginas seguidas** vienen 100%
   de inmuebles ya conocidos (sin cambios de precio), **corta** esa URL.
3. `ingest_master` hace *upsert* por `id_inmueble`: agrega nuevos, actualiza
   precios cambiados y refresca `last_seen`; conserva `first_seen`.

Resultado: la **primera** corrida es completa (siembra el master); las siguientes
solo tocan las primeras páginas nuevas → de **horas a minutos**.

```bash
python3 -m src.scraper --headless --incremental      # scrape (lee master, corta)
python3 -m src.ingest_master                          # upsert al master
python3 -m src.build_app_dataset                      # refresca el dashboard
```

**Sembrar el master con datos que ya tienes en `data/raw/`** (incluidos los CSV
viejos sin `id_inmueble`, que se deriva de la URL):

```bash
python3 -m src.ingest_master     # lee TODO data/raw y lo upserta al master
```

> El master es **liviano**: guarda solo las columnas útiles (descarta texto largo
> y campos basura), ~3–4 MB para decenas de miles de inmuebles.
>
> ⚠️ Detección de **bajas**: el corte temprano no re-confirma inmuebles viejos, así
> que las bajas (vendidos) no se detectan solas. Requiere un "barrido completo"
> periódico (sin `--incremental`) que marque inactivos — pendiente.

---

## Ingesta clásica y rastreo de cambios (opcional)

`src/ingest.py` es una vía alterna (sin master) que consolida todos los CSV:

```bash
python3 -m src.ingest     # → data/processed/housing_history.parquet + housing_clean.parquet
python3 -m src.changes    # 📈 cambios de precio · 🆕 nuevas · ❌ eliminadas (≥2 corridas)
```

---

## Dashboards

### `streamlit_app.py` — comercial (recomendado)

Dos vistas descriptivas para un equipo comercial inmobiliario. Lee
`data/app/housing_clean.parquet`.

- **📍 Precio por zona**: mapa por precio/m², ranking de barrios, KPIs.
- **⚖️ Comparador**: "esta propiedad vs. el mercado de su barrio" (percentil,
  precio sugerido, comparables).

```bash
streamlit run streamlit_app.py
```

Desplegable gratis en **Streamlit Community Cloud** (main file: `streamlit_app.py`).

### Otros

- `streamlit_dashboard.py` — autocontenido, entrena un modelo al vuelo.
- `src/app.py` — usa el modelo entrenado (`src/train.py` → `models/model.pkl`).

---

## Mapas interactivos (`src/viz_map.py`)

Genera un HTML interactivo (Folium) por ciudad, coloreado por precio/m². Lee el
dataset curado (rápido; funciona en CI sin CSV crudos).

```bash
python3 -m src.viz_map --ciudad Cali                       # → reports/cali_map.html
python3 -m src.viz_map --ciudad "Medellín" --out docs/medellin.html
```

**GitHub Pages**: la carpeta `docs/` (landing `index.html` + mapas) se publica en
`https://<usuario>.github.io/fr_scraper/`. El workflow regenera y deploya el mapa
con datos frescos en cada corrida (requiere **Settings → Pages → Source = "GitHub
Actions"**).

---

## Automatización (GitHub Actions)

[.github/workflows/scraper.yml](.github/workflows/scraper.yml) corre a diario (y a
demanda) en **tres etapas**:

1. **`scrape`** — matriz de **18 jobs por departamento** (`urls/<depto>.txt`), de a
   4 en paralelo, cada uno con `--incremental`. Cada job sube sus CSV como artifact.
2. **`merge`** — descarga todo, hace `ingest_master` (upsert al master) y
   `build_app_dataset`, y **commitea** `data/master` + `data/app` al repo (con
   `git pull --rebase --autostash` + reintento para evitar conflictos de push).
3. **`pages`** — regenera el mapa con datos frescos y lo **publica en GitHub Pages**.

Notas:
- El master vive en el repo para que los jobs lo lean; el **upsert es uno solo** en
  `merge` (los scrapers solo leen) → sin carreras.
- Límite de **6 h por job**: por eso se reparte por departamento. Con el master ya
  sembrado, las corridas son incrementales y rápidas.
- ⚠️ Scrapear a diario en Actions roza los ToS de FincaRaíz y las políticas de uso
  de GitHub. Mantén el paralelismo moderado (`max-parallel`).

---

## Almacenamiento del dato

Hoy el dato (`data/master`, `data/app`) vive **en el repo** como solución de MVP.
Esto genera fricción (commits del bot, conflictos de push) y no escala. Para
producción conviene mover el dato a **almacenamiento externo** — el `upsert` por
`id_inmueble` de `src/master.py` se traduce directo a un `MERGE` de SQL. Opciones
fáciles y con free tier: **Supabase/Neon (Postgres)**, **Cloudflare R2 / Backblaze
B2 (parquet)**, **MotherDuck (DuckDB)**, o **Snowflake** si el cliente ya lo usa.

---

## Notas y buenas prácticas

- **Respeta el sitio**: usa `--delay`/`--workers` moderados. El scraping puede
  violar los ToS de FincaRaíz; úsalo con fines educativos/de investigación y bajo
  tu responsabilidad. Para clientes, migra a datos propios/licenciados.
- FincaRaíz **cambia su HTML** cada tanto; si dejan de salir datos, revisa los
  selectores en `src/scraper.py` (`a.lc-data`, `.main-price`, `.lc-title`,
  `.lc-location`, `.lc-owner-name`, `div.lc-typologyTag`).
- Ejecuta **desde la raíz** del repo; en WSL/servidores usa `--headless`.

---

## Ideas de producto

[Guia.md](Guia.md) lista ideas de negocio/analítica sobre estos datos (dashboard
predictivo, informes de mercado, alertas de oportunidades, análisis espacial,
valoración para crédito, ranking) con un roadmap de MVP.
