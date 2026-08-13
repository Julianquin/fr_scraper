"""
Servicios cercanos desde OpenStreetMap — 100% offline, sin API ni límites.

Lee un extract de Colombia (.pbf de Geofabrik, en data/osm/) con DuckDB, extrae
los POIs relevantes y calcula features de "servicios cercanos" por inmueble
(conteos en radios + distancia a la más cercana de cada categoría).

Uso:
    # 1) (una vez) construir el parquet de POIs desde el .pbf
    python -m src.osm_pois

    # 2) en código / notebook
    from src.osm_pois import features_servicios
    df = features_servicios(df)   # añade columnas serv_*/n_*/dist_*
"""
from __future__ import annotations
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from src.config import BASE_DIR

OSM_DIR = BASE_DIR / "data" / "osm"
POIS_PARQUET = OSM_DIR / "pois_colombia.parquet"

# Categoría lógica → condición sobre los tags de OSM (nodos)
CATEGORIAS = {
    "educacion":    "tags['amenity'] IN ('school','university','college','kindergarten')",
    "salud":        "tags['amenity'] IN ('hospital','clinic','pharmacy','doctors')",
    "supermercado": "tags['shop'] IN ('supermarket','convenience','mall','marketplace')",
    "banco":        "tags['amenity'] IN ('bank','atm')",
    "gastronomia":  "tags['amenity'] IN ('restaurant','cafe','fast_food')",
    "parque":       "tags['leisure'] IN ('park','playground','pitch')",
    "transporte":   ("tags['public_transport'] IN ('station','stop_position') "
                     "OR tags['railway']='station' OR tags['amenity']='bus_station'"),
}


def extraer_pois(pbf: str | None = None, force: bool = False) -> pd.DataFrame:
    """Extrae los POIs del .pbf a un parquet cacheado (lat, lon, categoria)."""
    if POIS_PARQUET.exists() and not force:
        return pd.read_parquet(POIS_PARQUET)

    import duckdb
    if pbf is None:
        pbfs = sorted(glob.glob(str(OSM_DIR / "*.pbf")))
        if not pbfs:
            raise FileNotFoundError(f"No hay .pbf en {OSM_DIR} (descárgalo de Geofabrik).")
        pbf = pbfs[0]

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    conds = " OR ".join(f"({c})" for c in CATEGORIAS.values())
    cat_case = " ".join(f"WHEN {c} THEN '{n}'" for n, c in CATEGORIAS.items())
    q = f"""
        SELECT lat, lon, CASE {cat_case} END AS categoria
        FROM ST_ReadOSM('{pbf}')
        WHERE kind = 'node' AND lat IS NOT NULL AND ({conds})
    """
    df = con.execute(q).df().dropna(subset=["categoria"]).reset_index(drop=True)
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(POIS_PARQUET, index=False)
    print(f"✅ {len(df):,} POIs → {POIS_PARQUET}")
    print(df["categoria"].value_counts().to_string())
    return df


def features_servicios(listings: pd.DataFrame, radios_km=(0.5, 1.0)) -> pd.DataFrame:
    """Añade a cada inmueble: nº total de servicios por radio, nº por categoría a
    1 km y distancia (km) a la más cercana de cada categoría."""
    from sklearn.neighbors import BallTree

    pois = extraer_pois()
    out = listings.copy()
    coords = np.radians(out[["Latitud", "Longitud"]].values)

    tree_all = BallTree(np.radians(pois[["lat", "lon"]].values), metric="haversine")
    for r in radios_km:
        out[f"serv_{int(r*1000)}m"] = tree_all.query_radius(coords, r=r / 6371, count_only=True)

    for cat, g in pois.groupby("categoria"):
        t = BallTree(np.radians(g[["lat", "lon"]].values), metric="haversine")
        out[f"n_{cat}_1km"] = t.query_radius(coords, r=1 / 6371, count_only=True)
        out[f"dist_{cat}_km"] = t.query(coords, k=1)[0][:, 0] * 6371
    return out


if __name__ == "__main__":
    extraer_pois(force=True)
