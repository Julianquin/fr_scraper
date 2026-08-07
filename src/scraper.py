"""
Scraper de FincaRaiz → genera CSV en data/raw/.
Uso:
    python -m src.scraper --max-pages 30 --headless
"""

from __future__ import annotations
import argparse, logging, time, csv, concurrent.futures, functools
from pathlib import Path
from typing import Callable, Sequence, Dict, List

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from requests.adapters import HTTPAdapter, Retry
import json


# ───────────────────────────────────────── driver ──────────────────────────────────────────

def make_driver(*, headless: bool = True) -> webdriver.Chrome:
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")  # Chrome >= 109
    # acelerar: no cargar imágenes ni GPU
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    # Estabilidad en WSL/Linux headless: evita "Timed out receiving message
    # from renderer" por /dev/shm pequeño y problemas de sandbox.
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-extensions")
    opts.add_experimental_option(
        "prefs",
        {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2,
        },
    )
    opts.page_load_strategy = "eager"  # no esperar recursos secundarios
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=opts
    )
    driver.set_page_load_timeout(20)
    return driver

# ─────────────────────────────────── requests session util ─────────────────────────────────

def make_session(max_workers: int) -> requests.Session:
    """Sesión HTTP con keep‑alive y reintentos."""
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=max_workers * 2, pool_maxsize=max_workers * 2, max_retries=retries)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
    )
    return sess

# ───────────────────────────────────────── scraping utils ───────────────────────────────────
WAIT_SECS = 0  # más realista


def _parse_detail(soup_det: BeautifulSoup) -> Dict:
    """Extrae el diccionario de detalle a partir del soup ya renderizado."""
    detail: Dict[str, object] = {}

    # -- Información del proyecto (clave: valor) --
    for li in soup_det.select("div.project-info ul.ant-list-items li.ant-list-item"):
        cols = li.select("div.ant-col")
        if len(cols) >= 2:
            key = cols[0].get_text(strip=True)
            val = cols[1].get_text(strip=True)
            detail[key] = val

    # -- Descripción completa --
    desc = soup_det.select_one("div.property-description")
    detail["Descripción completa"] = desc.get_text(strip=True) if desc else None

    # -- Unidades / Tipos --
    units = []
    for unit_li in soup_det.select(
        "div.project-units-section ul.ant-list-items li.proyect_units_list_item"
    ):
        u: Dict[str, str] = {}
        for ui in unit_li.select("div.unit_item"):
            label = ui.select_one("strong").get_text(strip=True)
            value = ui.get_text(strip=True).replace(label, "").strip()
            u[label] = value
        units.append(u)
    detail["Unidades"] = units

    # -- Coordenadas desde JSON-LD --
    ld_json = soup_det.find("script", type="application/ld+json")
    if ld_json:
        try:
            data = json.loads(ld_json.string)
            geo = data.get("object", {}).get("geo", {})
            detail["Latitud"] = geo.get("latitude")
            detail["Longitud"] = geo.get("longitude")
        except Exception as e:
            detail["Error coordenadas"] = str(e)

    return detail


def scrape_detail(driver: webdriver.Chrome, wait: WebDriverWait, url: str) -> Dict:
    """Versión Selenium – se usa como *fallback* cuando la captación rápida falla."""
    driver.get(url)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1.property-title")))
    time.sleep(1)  # asegurar render final
    soup_det = BeautifulSoup(driver.page_source, "html.parser")
    return _parse_detail(soup_det)


def scrape_detail_fast(session: requests.Session, url: str) -> Dict:
    """Descarga HTML directamente alejándose de Selenium (mucho más rápido)."""
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    soup_det = BeautifulSoup(resp.text, "html.parser")
    return _parse_detail(soup_det)


def scrape_portal(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    url: str,
    *,
    workers: int = 8,
) -> List[Dict]:
    # 1) abrir listado con Selenium
    driver.get(url)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.listingCard")))
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    inmuebles = soup.select("div.listingCard")

    datos: List[Dict] = []
    detail_urls: List[str] = []

    def first(node, *selectors):
        """Primer elemento que matchee alguno de los selectores (soporta cambios de HTML)."""
        for s in selectors:
            el = node.select_one(s)
            if el:
                return el
        return None

    for itm in inmuebles:
        info: Dict[str, object] = {}
        # Enlace principal: nuevo HTML usa a.lc-data (antes a.lc-cardCover)
        link = first(itm, "a.lc-data", "a.lc-cardCover")
        if link:
            href = (link.get("href") or "").strip()
            info["Título"] = (link.get("title") or link.get_text(" ", strip=True) or "").strip()
            full_url = href if href.startswith("http") else f"https://www.fincaraiz.com.co{href}"
            info["URL detalle"] = full_url
            # El ID del inmueble es el último segmento numérico de la URL
            info["id_inmueble"] = full_url.rstrip("/").split("/")[-1] if href else None
            if href:
                detail_urls.append(full_url)
        img = itm.select_one("div.gallery-image img") or itm.select_one("img")
        info["URL imagen"] = (img.get("src") or "").strip() if img else None
        info["Etiquetas"] = [t.get_text(strip=True) for t in itm.select("span.property-tag")]
        precio = first(itm, "span.main-price", ".main-price", "span.price")
        info["Precio listado"] = precio.get_text(strip=True) if precio else None
        # La tipología viene en varios <span> (Habs / Baños / m²); tomamos el texto
        # completo del contenedor → "10 Habs. 4 Baños 432 m²" (formato que espera la limpieza)
        tip = itm.select_one("div.lc-typologyTag")
        info["Tipología listado"] = tip.get_text(" ", strip=True) if tip else None
        desc = first(itm, "span.lc-title", ".lc-title")
        info["Descripción breve"] = desc.get_text(strip=True) if desc else None
        loc = first(itm, "strong.lc-location", ".lc-location")
        info["Ubicación listado"] = loc.get_text(strip=True) if loc else None
        pub = first(itm, ".lc-owner-name", "div.publisher strong")
        info["Publicante"] = pub.get_text(strip=True) if pub else None
        btn = first(itm, ".btn-text", "div.property-lead-button button")
        info["Acción disponible"] = btn.get_text(strip=True) if btn else None
        datos.append(info)

    # 2) detalles en paralelo fuera de Selenium
    session = make_session(workers)
    detalles: Dict[str, Dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_url = {
            executor.submit(scrape_detail_fast, session, du): du for du in detail_urls
        }
        for fut in concurrent.futures.as_completed(future_to_url):
            url_det = future_to_url[fut]
            try:
                detalles[url_det] = fut.result()
            except Exception as exc:
                detalles[url_det] = {"Error detalle": str(exc)}

    # 3) merge + fallback Selenium si falló el scraping rápido
    for info in datos:
        du = info.get("URL detalle")
        if not du:  # sin URL no hay detalle que pedir (evita driver.get(None))
            continue
        detail_data = detalles.get(du, {})
        # si hubo error o datos muy incompletos → Selenium fallback
        if not detail_data or (
            "Descripción completa" not in detail_data and "Error detalle" not in detail_data
        ):
            try:
                detail_data = scrape_detail(driver, wait, du)
            except Exception as e:
                detail_data = {"Error detalle": str(e)}
        info.update(detail_data)

    return datos


ORDEN_RECIENTE = "ordenListado=3"  # orden por "más recientes" en FincaRaíz


def _page_url(base_url: str, page_num: int, *, orden: str | None = ORDEN_RECIENTE) -> str:
    """URL de la página N con el parámetro de orden (recientes) al final.
    Ej.: base/pagina2?ordenListado=3
    """
    path = base_url if page_num == 1 else f"{base_url}/pagina{page_num}"
    if not orden:
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}{orden}"


def scrape_multiple_pages(
    driver: webdriver.Chrome,
    wait: WebDriverWait,
    base_url: str,
    *,
    scraper: Callable[[webdriver.Chrome, WebDriverWait, str], Sequence[Dict]],
    max_pages: int = 50,
    stop_on_empty: bool = True,
    stop_on_error: bool = True,
    delay: float | None = None,
    known_prices: Dict[str, str] | None = None,
    stop_known_pages: int = 2,
) -> List[Dict]:
    logger = logging.getLogger("scraper")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s"))
        logger.addHandler(h)

    all_listings: List[Dict] = []
    known_streak = 0  # páginas consecutivas 100% ya conocidas (para corte incremental)

    for page_num in range(1, max_pages + 1):
        url = _page_url(base_url, page_num)
        logger.info("Scraping página %d: %s", page_num, url)

        try:
            page_data = list(scraper(driver, wait, url))
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en página %d → %s", page_num, exc)
            if stop_on_error:
                break
            continue

        if not page_data:
            logger.warning("Página %d sin resultados. Fin del scraping.", page_num)
            if stop_on_empty:
                break

        all_listings.extend(page_data)

        # ── Corte temprano incremental ──
        # Con orden por recientes, si una página no trae ningún inmueble nuevo ni
        # con precio cambiado, lo de abajo también es viejo → cortamos.
        if known_prices is not None and page_data:
            nuevos_o_cambiados = sum(
                1 for r in page_data
                if known_prices.get(str(r.get("id_inmueble"))) != str(r.get("Precio listado"))
            )
            if nuevos_o_cambiados == 0:
                known_streak += 1
                if known_streak >= stop_known_pages:
                    logger.info("Incremental: %d páginas seguidas ya conocidas → corto.",
                                known_streak)
                    break
            else:
                known_streak = 0

        if delay:
            time.sleep(delay)

    logger.info("Scraping finalizado. %d inmuebles recopilados.", len(all_listings))
    return all_listings

# ───────────────────────────────────────── CLI entrypoint ──────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-file", default="urls_fincaraiz.txt", help="archivo con URLs (una por línea)")
    parser.add_argument("--out-dir", default="data/raw", help="directorio destino CSV")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Vuelve a scrapear aunque el CSV exista")
    parser.add_argument("--workers", type=int, default=8, help="Hilos para detalles en paralelo")
    parser.add_argument("--recycle-every", type=int, default=25,
                        help="Reinicia Chrome cada N URLs para evitar fugas de memoria (0 = nunca)")
    parser.add_argument("--incremental", action="store_true",
                        help="Usa el master: orden por recientes + corta al llegar a lo ya conocido")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
    run_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    known_prices = None
    if args.incremental:
        from src import master as M
        known_prices = M.precios_conocidos()
        logging.info("🔁 Incremental ON — %d inmuebles conocidos (corta al llegar a lo conocido)",
                     len(known_prices))
    # Cada corrida guarda su snapshot en data/raw/<fecha>/ → nunca se pisa la historia
    out_dir = Path(args.out_dir) / run_date
    out_dir.mkdir(parents=True, exist_ok=True)

    driver = make_driver(headless=args.headless)
    wait = WebDriverWait(driver, WAIT_SECS)

    with open(args.url_file) as fh:
        urls = [u.strip() for u in fh if u.strip()]

    scraper_fn = functools.partial(scrape_portal, workers=args.workers)

    processed = 0
    for u in urls:
        # Nombre = ruta de la URL sin dominio. Funciona con cualquier profundidad:
        #   .../venta/bogota/bogota-dc            -> venta_bogota_bogota-dc
        #   .../venta/apartamentos/bogota/bogota-dc -> venta_apartamentos_bogota_bogota-dc
        path = u.split("//", 1)[-1].split("/", 1)[-1] if "//" in u else u
        fname = path.strip("/").replace("/", "_")
        csv_path = out_dir / f"{fname}.csv"

        if csv_path.exists() and not args.overwrite:
            logging.info("⏭️  %s ya existe (%s); omito scraping", fname, csv_path)
            continue

        # Reciclar Chrome cada N URLs para liberar memoria (evita timeouts del renderer)
        if processed and args.recycle_every and processed % args.recycle_every == 0:
            logging.info("♻️  Reiniciando Chrome tras %d URLs para liberar memoria", processed)
            try:
                driver.quit()
            except Exception:  # noqa: BLE001
                pass
            driver = make_driver(headless=args.headless)
            wait = WebDriverWait(driver, WAIT_SECS)
        processed += 1

        rows = scrape_multiple_pages(
            driver,
            wait,
            u,
            scraper=scraper_fn,
            max_pages=args.max_pages,
            delay=args.delay,
            known_prices=known_prices,
        )
        if not rows:
            continue

        df_out = pd.DataFrame(rows)
        df_out["fecha_recoleccion"] = run_date  # fecha en que se recolectó el dato
        df_out.to_csv(csv_path, index=False, encoding="utf-8")
        logging.info("✅ %d filas → %s", len(rows), csv_path)

    driver.quit()
    # El upsert del master se hace aparte con `python -m src.ingest_master`
    # (así el job "merge" lo hace una sola vez desde los CSV de todos los deptos).


if __name__ == "__main__":
    main()
