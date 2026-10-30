"""
Zonas oficiales de las ciudades (barrios y comunas o localidades), en data/geo/<ciudad>_{comunas,barrios}.geojson.

  - Cali: `descargar()` baja los polígonos del WFS de la IDESC (Alcaldía de Santiago de Cali).
  - Bogotá, Medellín, Barranquilla, Bucaramanga, Manizales, Armenia y Pereira: src/geo_ciudades.py (E6).
  - Se versionan simplificados: el CI no depende de que las fuentes estén arriba.
  - `asignar_zonas(df, ciudad)` le pone a cada aviso su barrio y su comuna (o localidad) oficial:
    por coordenadas (punto dentro del polígono) y, si no tiene, por nombre. Sin barrios (las ciudades
    con solo comunas de OSM), la comuna sale de su polígono. Las columnas se llaman `comuna` y
    `barrio_oficial` en todas las ciudades; CIUDADES dice cómo se llaman de verdad (comuna o localidad).
  - `canonica(ciudad)`: el nombre de la ciudad igual en los dos portales ("Bogotá D.C." → "Bogotá").

Uso:
    python -m src.zonas            # (re)descarga los polígonos de Cali a data/geo/
"""
from __future__ import annotations
import json
import re
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

# Cómo se llama la división principal, de dónde salen los polígonos y el transporte masivo (para el informe)
CIUDADES = {
    "Cali": {"division": "comuna", "fuente": "IDESC, Alcaldía de Cali", "transporte": "MIO"},
    "Bogotá": {"division": "localidad", "fuente": "Datos Abiertos Bogotá (SDP y UAECD)", "transporte": "TransMilenio"},
    "Medellín": {"division": "comuna", "fuente": "Catastro, Alcaldía de Medellín", "transporte": "Metro"},
    "Barranquilla": {"division": "localidad", "fuente": "POT 2024, Alcaldía de Barranquilla", "transporte": "Transmetro"},
    "Bucaramanga": {"division": "comuna", "fuente": "OpenStreetMap", "transporte": "Metrolínea"},
    "Manizales": {"division": "comuna", "fuente": "OpenStreetMap", "transporte": "transporte masivo"},
    "Armenia": {"division": "comuna", "fuente": "OpenStreetMap", "transporte": "transporte masivo"},
    "Pereira": {"division": "comuna", "fuente": "OpenStreetMap", "transporte": "Megabús"},
}
CANONICA = {"bogota d.c.": "Bogotá", "bogota, d.c.": "Bogotá", "bogota dc": "Bogotá",
            "cartagena de indias": "Cartagena"}

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


def slug(ciudad: str) -> str:
    """'Bogotá' → 'bogota' (prefijo de sus archivos en data/geo/)."""
    return normalizar(ciudad).replace(" ", "_")


def canonica(ciudad) -> str:
    """El nombre de la ciudad igual en los dos portales: Metrocuadrado escribe "Bogotá D.C." y
    "Cartagena de Indias"; FincaRaíz, "Bogotá" y "Cartagena"."""
    return CANONICA.get(normalizar(ciudad), ciudad) if isinstance(ciudad, str) else ciudad


def tiene_zonas(ciudad: str) -> bool:
    """Hay polígonos oficiales de esta ciudad en data/geo/."""
    return (GEO_DIR / f"{slug(ciudad)}_comunas.geojson").exists()


def division(ciudad: str) -> str:
    """'comuna' o 'localidad'."""
    return CIUDADES.get(ciudad, {}).get("division", "comuna")


def nombres(ciudad: str) -> dict[str, str]:
    """codigo de cada comuna o localidad → cómo se muestra: 'Comuna 2' (Cali), 'Chapinero' (Bogotá),
    'El Poblado (14)' (Medellín), 'Comuna 12 - Cabecera del Llano' (Bucaramanga)."""
    salida = {}
    for f in cargar("comunas", ciudad)["features"]:
        p = f["properties"]
        m = re.match(r"comuna\s+0*(\d+)$", p["nombre"], re.I)
        if m:
            salida[p["codigo"]] = f"Comuna {m.group(1)}"
        elif ciudad == "Medellín" and p.get("etiqueta") and int(p["etiqueta"]) < 50:
            salida[p["codigo"]] = f"{p['nombre']} ({p['etiqueta']})"
        else:
            salida[p["codigo"]] = p["nombre"]
    return salida


def etiquetas(ciudad: str) -> dict[str, str]:
    """codigo → la etiqueta corta del mapa (el número oficial; vacía si no lo hay)."""
    return {f["properties"]["codigo"]: f["properties"].get("etiqueta", str(int(f["properties"]["codigo"]))
                                                           if f["properties"]["codigo"].isdigit() else "")
            for f in cargar("comunas", ciudad)["features"]}


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


def cargar(nivel: str, ciudad: str = "Cali") -> dict:
    """GeoJSON de 'barrios' o 'comunas' de la ciudad desde data/geo/."""
    return json.loads((GEO_DIR / f"{slug(ciudad)}_{nivel}.geojson").read_text(encoding="utf-8"))


def asignar_zonas(df: pd.DataFrame, ciudad: str = "Cali") -> pd.DataFrame:
    """Añade `barrio_oficial`, `comuna` y `zona_por` ('coordenadas' | 'nombre').

    Las coordenadas mandan: el campo Barrio del aviso coincide con el barrio
    oficial solo ~40% de las veces. El nombre se usa solo si no hay coordenadas.
    Sin capa de barrios, la comuna sale directamente de su polígono.
    """
    df = df.copy()
    df["barrio_oficial"] = None
    df["comuna"] = None
    df["zona_por"] = None
    con_geo = df["Latitud"].notna() & df["Longitud"].notna()
    puntos = shapely.points(df.loc[con_geo, "Longitud"], df.loc[con_geo, "Latitud"])
    if not (GEO_DIR / f"{slug(ciudad)}_barrios.geojson").exists():
        feats = cargar("comunas", ciudad)["features"]
        i_pt, i_pol = shapely.STRtree([shape(f["geometry"]) for f in feats]).query(puntos, predicate="within")
        idx = df.index[con_geo][i_pt]
        df.loc[idx, "comuna"] = np.array([f["properties"]["codigo"] for f in feats], dtype=object)[i_pol]
        df.loc[idx, "zona_por"] = "coordenadas"
        return df

    feats = cargar("barrios", ciudad)["features"]
    geoms = [shape(f["geometry"]) for f in feats]
    nombres = np.array([f["properties"]["nombre"] for f in feats], dtype=object)
    comunas = np.array([f["properties"]["comuna"] for f in feats], dtype=object)

    # 1) Por coordenadas: punto dentro del polígono
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
