"""
Servicios cercanos desde OpenStreetMap: 100% offline, sin API ni límites (bloque E7).

Lee un extract de Colombia (.pbf de Geofabrik, en data/osm/) con DuckDB y guarda en
data/osm/pois_colombia.parquet un punto por servicio (lat, lon, categoria, area_m2).

Versión 2 (27-oct-2026). La primera solo leía NODOS (puntos sueltos) y se perdía casi todo lo que
OSM dibuja como polígono: en Cali quedaban 98 colegios y 28 parques. Ahora:
  - servicios: nodos + polígonos (centroide y área; el área separa un parque de una plazoleta);
  - vías principales y zonas industriales: se guardan TODOS sus nodos, para medir la distancia
    al borde y no al centro;
  - categorías que mueven el precio: transporte masivo (TransMilenio, Metro, MIO, cables…),
    centros comerciales, universidades, aeropuerto, comercio en general.

Uso:
    python -m src.osm_pois                       # (re)construye el parquet desde el .pbf (~1 min)
    python -m src.osm_pois --subir               # lo sube a R2 (osm/), de donde lo baja el entrenamiento
                                                 # semanal del CI (modelo.yml): el .pbf no sale de tu disco

    from src.osm_pois import features_servicios
    extra = features_servicios(df)               # SOLO las columnas nuevas: dist_*_km y n_*_500m
"""
from __future__ import annotations
import glob
import time

import numpy as np
import pandas as pd

from src.config import BASE_DIR

OSM_DIR = BASE_DIR / "data" / "osm"
POIS_PARQUET = OSM_DIR / "pois_colombia.parquet"
RADIO_TIERRA_KM = 6371.0
PARQUE_GRANDE_M2 = 10_000        # 1 ha: un parque de barrio de verdad, no una zona verde de conjunto
DIST_MAX_KM = 30.0               # más lejos que esto no aporta (y evita valores enormes en zonas rurales)

# Servicios: punto (nodo) o polígono (centroide + área). El orden importa: gana la primera que cumpla
CATEGORIAS = {
    "transporte_masivo": ("tags['public_transport']='station' OR tags['amenity']='bus_station' "
                          "OR tags['railway'] IN ('station','halt') OR tags['aerialway']='station'"),
    "parada_bus":        "tags['highway']='bus_stop'",
    "centro_comercial":  "tags['shop']='mall'",
    "supermercado":      "tags['shop']='supermarket'",
    "comercio":          "tags['shop'] IS NOT NULL",
    "gastronomia":       "tags['amenity'] IN ('restaurant','cafe','fast_food','bar','pub')",
    "banco":             "tags['amenity'] IN ('bank','atm')",
    "universidad":       "tags['amenity'] IN ('university','college')",
    "colegio":           "tags['amenity'] IN ('school','kindergarten')",
    "salud":             "tags['amenity'] IN ('hospital','clinic')",
    "parque":            "tags['leisure'] IN ('park','garden') OR tags['landuse']='recreation_ground'",
    "aeropuerto":        "tags['aeroway']='aerodrome'",
}
# Líneas y bordes: se guardan todos sus nodos (redondeados a ~10 m para no repetir)
LINEAS = {
    "via_principal": "tags['highway'] IN ('motorway','trunk','primary')",
    "industrial":    "tags['landuse']='industrial'",
}
# Qué se calcula para cada inmueble
DISTANCIAS = ["transporte_masivo", "centro_comercial", "supermercado", "universidad", "colegio", "salud",
              "parque", "parque_grande", "banco", "aeropuerto", "via_principal", "industrial"]
CONTEOS_500M = ["comercio", "gastronomia", "parada_bus", "colegio", "banco"]


def _caso(categorias: dict[str, str]) -> str:
    return "CASE " + " ".join(f"WHEN {c} THEN '{n}'" for n, c in categorias.items()) + " END"


def extraer_pois(pbf: str | None = None, force: bool = False) -> pd.DataFrame:
    """Extrae servicios, vías principales y zonas industriales del .pbf a un parquet cacheado."""
    if POIS_PARQUET.exists() and not force:
        return pd.read_parquet(POIS_PARQUET)
    import duckdb
    if pbf is None:
        pbfs = sorted(glob.glob(str(OSM_DIR / "*.pbf")))
        if not pbfs:
            raise FileNotFoundError(f"No hay .pbf en {OSM_DIR} (descárgalo de Geofabrik).")
        pbf = pbfs[-1]
    t0 = time.time()
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("SET geometry_always_xy = true")      # puntos en orden (longitud, latitud)
    cond_p = " OR ".join(f"({c})" for c in CATEGORIAS.values())
    cond_l = " OR ".join(f"({c})" for c in LINEAS.values())

    # 1) Nodos que son servicios
    nodos = con.execute(f"""
        SELECT lat, lon, {_caso(CATEGORIAS)} AS categoria, NULL::DOUBLE AS area_m2
        FROM ST_ReadOSM('{pbf}') WHERE kind = 'node' AND lat IS NOT NULL AND ({cond_p})
    """).df()

    # 2) Polígonos y líneas: sus nodos, en orden
    con.execute(f"""
        CREATE TEMP TABLE w AS
        SELECT id, refs,
               {_caso(LINEAS)} AS linea,
               {_caso(CATEGORIAS)} AS categoria
        FROM ST_ReadOSM('{pbf}') WHERE kind = 'way' AND (({cond_p}) OR ({cond_l}))
    """)
    con.execute("""CREATE TEMP TABLE wr AS
                   SELECT id, unnest(refs) AS nid, generate_subscripts(refs, 1) AS pos FROM w""")
    con.execute(f"""CREATE TEMP TABLE n AS
                    SELECT o.id, o.lat, o.lon FROM ST_ReadOSM('{pbf}') o
                    WHERE o.kind = 'node' AND o.id IN (SELECT DISTINCT nid FROM wr)""")
    poligonos = con.execute("""
        WITH c AS (
            SELECT w.id, any_value(w.categoria) AS categoria, any_value(w.refs) AS refs,
                   list(ST_Point(n.lon, n.lat) ORDER BY wr.pos) AS puntos,
                   avg(n.lat) AS lat, avg(n.lon) AS lon
            FROM w JOIN wr ON wr.id = w.id JOIN n ON n.id = wr.nid
            WHERE w.linea IS NULL GROUP BY w.id
        )
        SELECT lat, lon, categoria,
               CASE WHEN refs[1] = refs[-1] AND len(puntos) >= 4
                    THEN TRY(ST_Area_Spheroid(ST_MakePolygon(ST_MakeLine(puntos)))) END AS area_m2
        FROM c
    """).df()
    lineas = con.execute("""
        SELECT DISTINCT round(n.lat, 4) AS lat, round(n.lon, 4) AS lon, w.linea AS categoria, NULL::DOUBLE AS area_m2
        FROM w JOIN wr ON wr.id = w.id JOIN n ON n.id = wr.nid WHERE w.linea IS NOT NULL
    """).df()

    df = pd.concat([nodos, poligonos, lineas], ignore_index=True).dropna(subset=["categoria", "lat", "lon"])
    df["area_m2"] = df["area_m2"].abs()
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(POIS_PARQUET, index=False)
    print(f"✅ {len(df):,} puntos → {POIS_PARQUET.relative_to(BASE_DIR)} ({time.time() - t0:.0f} s)")
    print(df["categoria"].value_counts().to_string())
    return df


def features_servicios(df: pd.DataFrame, pois: pd.DataFrame | None = None) -> pd.DataFrame:
    """Para cada inmueble: distancia (km) al más cercano de cada categoría de DISTANCIAS y cuántos
    hay a 500 m de cada una de CONTEOS_500M. Sin coordenadas → vacío."""
    from sklearn.neighbors import BallTree

    pois = extraer_pois() if pois is None else pois
    lat, lon = ("latitud", "longitud") if "latitud" in df else ("Latitud", "Longitud")
    ok = df[lat].notna() & df[lon].notna()
    coords = np.radians(df.loc[ok, [lat, lon]].astype(float).to_numpy())
    out = pd.DataFrame(index=df.index)
    grupos = dict(tuple(pois.groupby("categoria")))
    grupos["parque_grande"] = pois[(pois["categoria"] == "parque") & (pois["area_m2"] >= PARQUE_GRANDE_M2)]
    for cat in DISTANCIAS:
        g = grupos.get(cat)
        if g is None or g.empty:
            continue
        arbol = BallTree(np.radians(g[["lat", "lon"]].to_numpy()), metric="haversine")
        d = arbol.query(coords, k=1)[0][:, 0] * RADIO_TIERRA_KM
        out.loc[ok, f"dist_{cat}_km"] = np.minimum(d, DIST_MAX_KM)
    for cat in CONTEOS_500M:
        g = grupos.get(cat)
        if g is None or g.empty:
            continue
        arbol = BallTree(np.radians(g[["lat", "lon"]].to_numpy()), metric="haversine")
        out.loc[ok, f"n_{cat}_500m"] = arbol.query_radius(coords, r=0.5 / RADIO_TIERRA_KM, count_only=True)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Servicios cercanos desde OpenStreetMap")
    p.add_argument("--subir", action="store_true",
                   help="sube el parquet a R2 (osm/pois_colombia.parquet); lo construye solo si no existe")
    a = p.parse_args()
    if a.subir:
        from src import almacen
        extraer_pois()
        almacen.subir_archivo(POIS_PARQUET, f"osm/{POIS_PARQUET.name}")
    else:
        extraer_pois(force=True)
