"""
Scraper nuevo de FincaRaíz por HTTP (bloque E1). Corre EN PARALELO al scraper actual y NO toca
el master, la bitácora ni el dataset activo: escribe solo su propio crudo.

Incremental diario: recorre /venta (todo el país) ordenado por "más recientes" y se queda con los
avisos publicados desde `--desde` (por defecto, ayer en Colombia). Para cuando encuentra 2 páginas
seguidas con avisos más viejos, al llegar al tope del buscador (10.000 resultados, página 476) o
al llegar a `--max-paginas`.

Salidas:
    data/raw_http/<fecha>/venta.parquet                 local (git lo ignora)
    raw/venta_http/fecha=<fecha>/scrape.parquet         en R2, con --subir

Uso:
    python -m src.scraper_http                          # avisos desde ayer
    python -m src.scraper_http --desde 2026-10-20 --subir
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import BASE_DIR
from src.fincaraiz_http import ULTIMA_PAGINA, aplanar, pedir

OUT_DIR = BASE_DIR / "data" / "raw_http"
PAGINAS_VIEJAS_PARA_CORTAR = 2


def incremental(desde: str, max_paginas: int = ULTIMA_PAGINA, pausa: float = 1.0) -> pd.DataFrame:
    """Avisos publicados desde `desde` (AAAA-MM-DD), leyendo /venta del más reciente al más viejo."""
    sesion = requests.Session()
    filas, previos, viejas, repetidas, motivo = [], None, 0, 0, "llegó al máximo de páginas"
    t0 = time.time()
    # El tope del buscador lo controla ULTIMA_PAGINA. Una página idéntica a la anterior no es
    # señal de tope: la caché del portal a veces sirve la misma página para dos URLs distintas.
    for n in range(1, min(max_paginas, ULTIMA_PAGINA) + 1):
        data, info, _ = pedir(sesion, "venta", n)
        ids = [str(a.get("id")) for a in data]
        if not data:
            motivo = f"página {n} vacía"
            break
        repetidas += ids == previos
        previos = ids
        fechas = [a.get("created_at") or "" for a in data]
        filas += [aplanar(a) | {"pagina": n} for a, f in zip(data, fechas) if f >= desde]
        viejas = viejas + 1 if all(f < desde for f in fechas) else 0
        if viejas >= PAGINAS_VIEJAS_PARA_CORTAR:
            motivo = f"{viejas} páginas seguidas con avisos anteriores a {desde}"
            break
        time.sleep(pausa)

    df = pd.DataFrame(filas)
    if df.empty:
        print(f"Sin avisos desde {desde} ({motivo})")
        return df
    antes = len(df)
    df = df.drop_duplicates("id_inmueble", keep="first").reset_index(drop=True)   # el portal repite avisos entre páginas
    df["fecha_recoleccion"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    print(f"{n} páginas en {(time.time() - t0) / 60:.1f} min · corte: {motivo} · "
          f"páginas idénticas a la anterior (caché del portal): {repetidas}")
    print(f"{len(df):,} avisos únicos publicados desde {desde} ({antes - len(df)} repetidos descartados) · "
          f"total nacional declarado: {info.get('total'):,}")
    return df


def guardar(df: pd.DataFrame, fecha: str, subir: bool) -> Path:
    destino = OUT_DIR / fecha / "venta.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    from src.almacen import COMPRESION
    df.to_parquet(destino, index=False, compression=COMPRESION)
    print(f"💾 {destino.relative_to(BASE_DIR)} ({destino.stat().st_size / 1e6:.1f} MB)")
    if subir:
        from src import almacen
        almacen.subir_archivo(destino, f"raw/venta_http/fecha={fecha}/scrape.parquet")
    return destino


def resumen(df: pd.DataFrame) -> None:
    print("Por fecha de publicación:", df["fecha_publicacion"].value_counts().sort_index().to_dict())
    print("Ciudades principales:", df["ciudad"].value_counts().head(8).to_dict())
    master = BASE_DIR / "data" / "master" / "listings.parquet"
    if master.exists():
        ids = set(pd.read_parquet(master, columns=["id_inmueble"])["id_inmueble"].astype(str))
        nuevos = ~df["id_inmueble"].isin(ids)
        print(f"No están en el master actual: {nuevos.sum():,} de {len(df):,} ({nuevos.mean():.0%})")


if __name__ == "__main__":
    ayer = (pd.Timestamp.now(tz="America/Bogota") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="Incremental de /venta por HTTP (en paralelo al scraper actual)")
    p.add_argument("--desde", default=ayer, help="fecha de publicación mínima (AAAA-MM-DD); por defecto, ayer")
    p.add_argument("--max-paginas", type=int, default=ULTIMA_PAGINA)
    p.add_argument("--pausa", type=float, default=1.0, help="segundos entre páginas (cortesía)")
    p.add_argument("--subir", action="store_true", help="sube el crudo a R2 (raw/venta_http/)")
    a = p.parse_args()

    df = incremental(a.desde, a.max_paginas, a.pausa)
    if not df.empty:
        guardar(df, df["fecha_recoleccion"].iloc[0], a.subir)
        resumen(df)
