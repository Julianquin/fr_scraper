"""
Zonas oficiales de las ciudades (barrios y comunas o localidades), en data/geo/<ciudad>_{comunas,barrios}.geojson.

  - Cali: `descargar()` baja los polígonos del WFS de la IDESC (Alcaldía de Santiago de Cali).
  - Bogotá, Medellín, Barranquilla, Cartagena, Bucaramanga, Manizales, Armenia y Pereira: src/geo_ciudades.py (E6).
  - Se versionan simplificados: el CI no depende de que las fuentes estén arriba.
  - `asignar_zonas(df, ciudad)` le pone a cada aviso su barrio y su comuna (o localidad) oficial:
    por coordenadas (punto dentro del polígono) y, si no tiene, por nombre. Sin barrios (las ciudades
    con solo comunas de OSM), la comuna sale de su polígono. Las columnas se llaman `comuna` y
    `barrio_oficial` en todas las ciudades; CIUDADES dice cómo se llaman de verdad (comuna o localidad).
  - `canonica(ciudad)`: el nombre de la ciudad igual en los dos portales ("Bogotá D.C." → "Bogotá").
  - Áreas metropolitanas (AREAS): el núcleo con sus comunas o localidades y barrios, su parte rural y cada
    municipio vecino como una zona (límites municipales del DANE). Los avisos entran por sus coordenadas
    dentro del área (`en_area`), no por el texto del portal: "Medellín" en un aviso de Envigado no engaña.

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
    "Cartagena": {"division": "localidad", "fuente": "Catastro Distrital de Cartagena", "transporte": "Transcaribe"},
}
# Áreas metropolitanas: núcleo y municipios vecinos (nombre y departamento como los escribe el DANE)
AREAS = {
    "Valle de Aburrá": ("Medellín", "Antioquia", ["Bello", "Itagüí", "Envigado", "Sabaneta", "La Estrella", "Caldas",
                                                  "Copacabana", "Girardota", "Barbosa"]),
    "Área metropolitana de Barranquilla": ("Barranquilla", "Atlántico", ["Soledad", "Malambo", "Puerto Colombia",
                                                                         "Galapa"]),
    "Área metropolitana de Bucaramanga": ("Bucaramanga", "Santander", ["Floridablanca", "Girón", "Piedecuesta"]),
    "Área metropolitana de Pereira": ("Pereira", "Risaralda", ["Dosquebradas", "La Virginia"]),
    "Bogotá y la Sabana": ("Bogotá", "Cundinamarca", ["Chía", "Cajicá", "Cota", "Funza", "Mosquera", "Madrid", "Soacha",
                                                     "La Calera", "Sopó", "Zipaquirá", "Tenjo", "Tabio", "Tocancipá"]),
    "Cali y alrededores": ("Cali", "Valle del Cauca", ["Jamundí", "Yumbo", "Palmira", "Candelaria"]),
}
for _area, (_nucleo, _, _) in AREAS.items():
    CIUDADES[_area] = {"division": "zona", "transporte": CIUDADES[_nucleo]["transporte"],
                       "fuente": CIUDADES[_nucleo]["fuente"] + "; municipios: DANE (MGN, vía HDX)"}
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
CERCA = 0.0018         # ~200 m: un aviso fuera de toda zona pero a menos de esto va a la más cercana (la costa
                       # de Cartagena, los bordes entre la fuente oficial y el DANE)


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


def nucleo(ciudad: str) -> str:
    """La ciudad central de un área metropolitana (o la misma ciudad)."""
    return AREAS[ciudad][0] if ciudad in AREAS else ciudad


_POLIGONO: dict = {}


def en_area(area: str, lat, lon, texto=None) -> np.ndarray:
    """Qué avisos son del área metropolitana: los que caen dentro de sus zonas (data/geo/) y, sin coordenadas,
    los que dicen uno de sus municipios (`texto`, la ciudad del aviso)."""
    if area not in _POLIGONO:
        _POLIGONO[area] = shapely.union_all([shape(f["geometry"]) for f in cargar("comunas", area)["features"]])
        shapely.prepare(_POLIGONO[area])
    lat, lon = pd.to_numeric(pd.Series(lat), errors="coerce"), pd.to_numeric(pd.Series(lon), errors="coerce")
    dentro = np.zeros(len(lat), dtype=bool)
    ok = (lat.notna() & lon.notna()).to_numpy()
    dentro[ok] = shapely.contains_xy(_POLIGONO[area], lon.to_numpy()[ok], lat.to_numpy()[ok])
    if texto is not None:
        nombres = {normalizar(AREAS[area][0]), *map(normalizar, AREAS[area][2])}
        dentro |= ~ok & pd.Series(texto).map(canonica).map(normalizar).isin(nombres).to_numpy()
    return dentro


def archivo(nivel: str, ciudad: str) -> Path:
    """data/geo/<ciudad>_<nivel>.geojson; los barrios de un área son los de su núcleo."""
    return GEO_DIR / f"{slug(nucleo(ciudad) if nivel == 'barrios' else ciudad)}_{nivel}.geojson"


def tiene_zonas(ciudad: str) -> bool:
    """Hay polígonos oficiales de esta ciudad en data/geo/."""
    return archivo("comunas", ciudad).exists()


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
        elif nucleo(ciudad) == "Medellín" and p.get("etiqueta") and int(p["etiqueta"]) < 50:
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
    return json.loads(archivo(nivel, ciudad).read_text(encoding="utf-8"))


def asignar_zonas(df: pd.DataFrame, ciudad: str = "Cali") -> pd.DataFrame:
    """Añade `barrio_oficial`, `comuna` y `zona_por` ('coordenadas' | 'nombre' | 'cercanía').

    Las coordenadas mandan: el campo Barrio del aviso coincide con el barrio
    oficial solo ~40% de las veces. El nombre se usa solo si no hay coordenadas.
    Sin capa de barrios, la comuna sale directamente de su polígono. Lo que quede
    fuera de toda zona a menos de CERCA va a la comuna más cercana; más lejos,
    sin zona (casi siempre, otro municipio: Yumbo, Puerto Colombia…).
    """
    df = df.copy()
    df["barrio_oficial"] = None
    df["comuna"] = None
    df["zona_por"] = None
    con_geo = df["Latitud"].notna() & df["Longitud"].notna()
    puntos = shapely.points(df.loc[con_geo, "Longitud"], df.loc[con_geo, "Latitud"])
    if not archivo("barrios", ciudad).exists():
        feats = cargar("comunas", ciudad)["features"]
        i_pt, i_pol = shapely.STRtree([shape(f["geometry"]) for f in feats]).query(puntos, predicate="within")
        idx = df.index[con_geo][i_pt]
        df.loc[idx, "comuna"] = np.array([f["properties"]["codigo"] for f in feats], dtype=object)[i_pol]
        df.loc[idx, "zona_por"] = "coordenadas"
        return _cercanos(df, ciudad)

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
    por_nombre = dict(zip(map(normalizar, nombres[::-1]), zip(nombres[::-1], comunas[::-1])))  # gana el primero (urbano)
    sin_geo = ~con_geo & df["Barrio"].notna()
    match = df.loc[sin_geo, "Barrio"].map(lambda b: por_nombre.get(normalizar(b)))
    match = match.dropna()
    df.loc[match.index, "barrio_oficial"] = match.str[0]
    df.loc[match.index, "comuna"] = match.str[1]
    df.loc[match.index, "zona_por"] = "nombre"

    # 3) Sin comuna (fuera de los barrios, o en un municipio vecino del área): por el polígono de la comuna
    falta = df["comuna"].isna() & con_geo
    if falta.any():
        feats = cargar("comunas", ciudad)["features"]
        pts = shapely.points(df.loc[falta, "Longitud"], df.loc[falta, "Latitud"])
        i_pt, i_pol = shapely.STRtree([shape(f["geometry"]) for f in feats]).query(pts, predicate="within")
        idx = df.index[falta][i_pt]
        df.loc[idx, "comuna"] = np.array([f["properties"]["codigo"] for f in feats], dtype=object)[i_pol]
        df.loc[idx, "zona_por"] = df.loc[idx, "zona_por"].fillna("coordenadas")
    return _cercanos(df, ciudad)


def _cercanos(df: pd.DataFrame, ciudad: str) -> pd.DataFrame:
    """4) Fuera de toda zona pero a menos de CERCA: la comuna más cercana (zona_por 'cercanía')."""
    falta = df["comuna"].isna() & df["Latitud"].notna() & df["Longitud"].notna()
    if falta.any():
        feats = cargar("comunas", ciudad)["features"]
        pts = shapely.points(df.loc[falta, "Longitud"], df.loc[falta, "Latitud"])
        i_pt, i_pol = shapely.STRtree([shape(f["geometry"]) for f in feats]).query_nearest(
            pts, max_distance=CERCA, all_matches=False)
        idx = df.index[falta][i_pt]
        df.loc[idx, "comuna"] = np.array([f["properties"]["codigo"] for f in feats], dtype=object)[i_pol]
        df.loc[idx, "zona_por"] = "cercanía"
    return df


if __name__ == "__main__":
    descargar()
