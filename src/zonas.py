"""
Zonas oficiales de Cali (barrios y comunas) desde la IDESC — Alcaldía de Santiago de Cali.

  - `descargar()` baja los polígonos del WFS de la IDESC, los simplifica y los
    guarda en data/geo/ (se versionan: el CI no depende de que la IDESC esté arriba).
  - `asignar_zonas(df)` le pone a cada aviso su barrio y comuna oficial:
    por coordenadas (punto dentro del polígono) y, si no tiene, por nombre.

Uso:
    python -m src.zonas            # (re)descarga los polígonos a data/geo/
"""
from __future__ import annotations
import json
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape, mapping

from src.config import BASE_DIR

GEO_DIR = BASE_DIR / "data" / "geo"
WFS = ("https://ws-idesc.cali.gov.co/geoserver/wfs?service=WFS&version=2.0.0"
       "&request=GetFeature&outputFormat=application/json&srsName=EPSG:4326&typeNames=")
FUENTE = "IDESC – Alcaldía de Santiago de Cali (Acuerdo 0636 de 2026)"

# capa WFS → (archivo, {campo IDESC: campo nuestro})
CAPAS = {
    "dapm:pdt_dpa_barrios_sectores": ("cali_barrios.geojson",
                                      {"barcodigo": "codigo", "barnombre": "nombre",
                                       "barcomuna": "comuna"}),
    "dapm:pdt_dpa_comunas": ("cali_comunas.geojson",
                             {"comcodigo": "codigo", "comnombre": "nombre"}),
}
TOLERANCIA = 0.00005   # ~5 m: simplificación de bordes para aligerar el HTML


def normalizar(s) -> str:
    """Minúsculas, sin tildes ni espacios sobrantes (para emparejar nombres)."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().split())


def descargar() -> None:
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    for capa, (archivo, campos) in CAPAS.items():
        with urllib.request.urlopen(WFS + capa, timeout=180) as r:
            gj = json.load(r)
        feats = []
        for f in gj["features"]:
            geom = shapely.set_precision(shape(f["geometry"]).simplify(TOLERANCIA), 1e-5)
            props = {nuevo: f["properties"][viejo] for viejo, nuevo in campos.items()}
            feats.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
        out = GEO_DIR / archivo
        out.write_text(json.dumps({"type": "FeatureCollection", "fuente": FUENTE,
                                   "features": feats}, ensure_ascii=False), encoding="utf-8")
        print(f"✅ {len(feats)} polígonos → {out} ({out.stat().st_size / 1e3:.0f} KB)")


def cargar(nivel: str) -> dict:
    """GeoJSON de 'barrios' o 'comunas' desde data/geo/."""
    return json.loads((GEO_DIR / f"cali_{nivel}.geojson").read_text(encoding="utf-8"))


def asignar_zonas(df: pd.DataFrame) -> pd.DataFrame:
    """Añade `barrio_oficial`, `comuna` y `zona_por` ('coordenadas' | 'nombre').

    Las coordenadas mandan: el campo Barrio del aviso coincide con el barrio
    oficial solo ~40% de las veces. El nombre se usa solo si no hay coordenadas.
    """
    feats = cargar("barrios")["features"]
    geoms = [shape(f["geometry"]) for f in feats]
    nombres = np.array([f["properties"]["nombre"] for f in feats], dtype=object)
    comunas = np.array([f["properties"]["comuna"] for f in feats], dtype=object)

    df = df.copy()
    df["barrio_oficial"] = None
    df["comuna"] = None
    df["zona_por"] = None

    # 1) Por coordenadas: punto dentro del polígono
    con_geo = df["Latitud"].notna() & df["Longitud"].notna()
    puntos = shapely.points(df.loc[con_geo, "Longitud"], df.loc[con_geo, "Latitud"])
    i_pt, i_pol = shapely.STRtree(geoms).query(puntos, predicate="within")
    idx = df.index[con_geo][i_pt]
    df.loc[idx, "barrio_oficial"] = nombres[i_pol]
    df.loc[idx, "comuna"] = comunas[i_pol]
    df.loc[idx, "zona_por"] = "coordenadas"

    # 2) Sin coordenadas: por nombre normalizado
    por_nombre = dict(zip(map(normalizar, nombres), zip(nombres, comunas)))
    sin_geo = ~con_geo & df["Barrio"].notna()
    match = df.loc[sin_geo, "Barrio"].map(lambda b: por_nombre.get(normalizar(b)))
    match = match.dropna()
    df.loc[match.index, "barrio_oficial"] = match.str[0]
    df.loc[match.index, "comuna"] = match.str[1]
    df.loc[match.index, "zona_por"] = "nombre"
    return df


if __name__ == "__main__":
    descargar()
