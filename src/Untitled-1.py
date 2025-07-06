# src/scraper_test.py
"""
Scraper de FincaRaiz → genera CSV en data/raw/.
Uso:
    python -m src.scraper --max-pages 30 --headless
"""

from __future__ import annotations
import argparse, logging, time, csv
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


# ───────────────────────────────────────── driver ──────────────────────────────────────────
def make_driver(*, headless: bool = True) -> webdriver.Chrome:
    opts = webdriver.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")  # Chrome >= 109
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

# ───────────────────────────────────────── scraping utils ───────────────────────────────────
WAIT_SECS = 0

def scrape_detail(driver: webdriver.Chrome, wait: WebDriverWait, url: str) -> Dict:
    driver.get(url)
    # 1) Esperar a que cargue el título del detalle
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.property-title')))
    time.sleep(1)  # opcional: asegurar que toda la sección renderice

    soup_det = BeautifulSoup(driver.page_source, 'html.parser')
    detail = {}

    # -- Información del proyecto (clave: valor) --
    for li in soup_det.select('div.project-info ul.ant-list-items li.ant-list-item'):
        cols = li.select('div.ant-col')
        if len(cols) >= 2:
            key = cols[0].get_text(strip=True)
            val = cols[1].get_text(strip=True)
            detail[key] = val

    # -- Descripción completa --
    desc = soup_det.select_one('div.property-description')
    detail['Descripción completa'] = desc.get_text(strip=True) if desc else None

    # -- Unidades / Tipos --
    units = []
    for unit_li in soup_det.select('div.project-units-section ul.ant-list-items li.proyect_units_list_item'):
        u = {}
        for ui in unit_li.select('div.unit_item'):
            label = ui.select_one('strong').get_text(strip=True)
            # el texto restante tras el <strong>
            value = ui.get_text(strip=True).replace(label, '').strip()
            u[label] = value
        units.append(u)
    detail['Unidades'] = units

    return detail

def scrape_portal(driver: webdriver.Chrome, wait: WebDriverWait, url: str) -> List[Dict]:
    driver.get(url)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.listingCard')))
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    inmuebles = soup.select('div.listingCard')

    datos = []
    for itm in inmuebles:
        info = {}
        # -- Resumen en listado (igual que antes) --
        cover = itm.select_one('a.lc-cardCover')
        if cover:
            info['Título'] = cover.get('title', '').strip()
            href = cover.get('href', '').strip()
            info['URL detalle'] = f"https://www.fincaraiz.com.co{href}"
        img = itm.select_one('img.card-image-gallery--img')
        info['URL imagen'] = img['src'].strip() if img else None
        info['Etiquetas'] = [t.get_text(strip=True) for t in itm.select('span.property-tag')]
        precio = itm.select_one('span.price')
        info['Precio listado'] = precio.get_text(strip=True) if precio else None
        tip = itm.select_one('div.lc-typologyTag span')
        info['Tipología listado'] = tip.get_text(' ', strip=True) if tip else None
        desc = itm.select_one('span.lc-title')
        info['Descripción breve'] = desc.get_text(strip=True) if desc else None
        loc = itm.select_one('strong.lc-location')
        info['Ubicación listado'] = loc.get_text(strip=True) if loc else None
        pub = itm.select_one('div.publisher strong')
        info['Publicante'] = pub.get_text(strip=True) if pub else None
        btn = itm.select_one('div.property-lead-button button')
        info['Acción disponible'] = btn.get_text(strip=True) if btn else None

        # -- Scrape de la página de detalle y fusión --
        if info.get('URL detalle'):
            try:
                det = scrape_detail(info['URL detalle'])
                info.update(det)
            except Exception as e:
                info['Error detalle'] = str(e)

        datos.append(info)

    return datos

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
) -> List[Dict]:
    logger = logging.getLogger("scraper")
    if not logger.handlers:                           # ← evita duplicados
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s │ %(levelname)s │ %(message)s"))
        logger.addHandler(h)


    all_listings: List[Dict] = []

    for page_num in range(1, max_pages + 1):
        url = base_url if page_num == 1 else f"{base_url}/pagina{page_num}"
        logger.info("Scraping página %d: %s", page_num, url)

        try:
            page_data = list(scraper(driver, wait, url))

        except Exception as exc:             # noqa: BLE001  (cazar la excepción base es OK aquí)
            logger.error("Error en página %d → %s", page_num, exc)
            if stop_on_error:
                break
            continue

        if not page_data:
            logger.warning("Página %d sin resultados. Fin del scraping.", page_num)
            if stop_on_empty:
                break

        all_listings.extend(page_data)

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
    parser.add_argument("--overwrite", action="store_true",   # ⬅️  NUEVO
                    help="Vuelve a scrapear aunque el CSV exista")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    driver = make_driver(headless=args.headless)
    wait = WebDriverWait(driver, WAIT_SECS)

    with open(args.url_file) as fh:
        urls = [u.strip() for u in fh if u.strip()]

    for u in urls:
        
        parts   = u.split("/")
        fname   = "_".join(parts[-4:-1])          # venta_bogota_bogota-dc
        csv_path = out_dir / f"{fname}.csv"
        # ── Skip si el fichero ya existe ───────────────────────────────────────
        if csv_path.exists() and not args.overwrite:
            logging.info("⏭️  %s ya existe (%s); omito scraping", fname, csv_path)
            continue
        # ----------------------------------------------------------------------
        rows = scrape_multiple_pages(driver, wait, u,scraper=scrape_portal, max_pages=args.max_pages, delay=args.delay)
        if not rows:
            continue

        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
        logging.info("✅ %d filas → %s", len(rows), csv_path)

    driver.quit()

if __name__ == "__main__":
    main()
# python -m src.scraper --headless --max-pages 50 --overwrite