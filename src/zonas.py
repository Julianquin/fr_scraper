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
COLOMBIA = (-4.3, 13.6, -82.0, -66.8)   # latitud y longitud mínima y máxima (con San Andrés)
LEJOS_BARRIO = 3       # km: más lejos que esto del barrio que nombra, y fuera de su ciudad, el punto está mal
LEJOS_CIUDAD = 10      # km: más lejos que esto de su ciudad, el punto es de otro sitio (Bogotá para un aviso de Cali)


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


_UBICACION: dict = {}
# Palabras que no distinguen un lugar: se quitan antes de comparar nombres
VACIAS = {"el", "la", "los", "las", "de", "del", "y", "barrio", "urbanizacion", "urb", "sector", "conjunto",
          "condominio", "corregimiento", "comuna", "zona", "vereda", "parcelacion", "na", "n", "a"}
# Nombres de una palabra demasiado comunes para emparejar solo por estar contenidos en otro
GENERICAS = {"ciudad", "villa", "san", "santa", "nueva", "nuevo", "jardin", "jardines", "parque", "alto", "altos",
             "bosque", "bosques", "campestre", "country", "club", "real", "reserva", "portal", "torres", "colinas",
             "lomas", "brisas", "prados", "valle", "centro", "norte", "sur", "oeste", "oriente", "occidente", "rural",
             "expansion", "cabecera", "industrial", "residencial", "etapa", "manzana", "mz", "casa", "lote"}


def clave(nombre) -> str:
    """'Urbanización La Nueva Base' → 'nueva base'; 'Pance (cabecera)' → 'pance' (para emparejar nombres)."""
    s = re.sub(r"\(.*?\)", " ", normalizar(nombre))
    return " ".join(t for t in re.sub(r"[^a-z0-9 ]", " ", s).split() if t not in VACIAS)


def _alias(nombre: str) -> list[str]:
    """Las claves de un nombre y de cada parte de un nombre compuesto ('Tejares - Cristales' → 3 claves)."""
    partes = [nombre] + (nombre.split(" - ") if " - " in nombre else [])
    return [k for k in dict.fromkeys(clave(x) for x in partes) if k]


def _coinciden(k: str, c: str) -> bool:
    """La clave `c` es `k`, o una contiene a la otra palabra por palabra (no vale una sola palabra genérica)."""
    if c == k:
        return True
    unica = lambda x: " " not in x and x in GENERICAS
    return len(c) >= 4 and not unica(c) and not unica(k) and (f" {c} " in f" {k} " or f" {k} " in f" {c} ")


def _comparten(k: str, c: str) -> int:
    return len(set(k.split()) & set(c.split()))


def _buscar(k: str, oficiales: list, tablas: list[dict]):
    """El punto del lugar que nombra `k`. Primero los barrios oficiales (`oficiales`: [(clave, polígono, punto)])
    que coincidan, el que más palabras comparta y, a igualdad, el más pequeño (el urbano antes que la vereda:
    "Pance" → Parcelaciones Pance y no la cabecera del corregimiento); si no hay, la primera de `tablas`
    (clave → punto) con alguno: la misma clave o la que más palabras comparta."""
    if not k:
        return None
    cand = [(c, g, pt) for c, g, pt in oficiales if _coinciden(k, c)]
    if cand:
        return max(cand, key=lambda o: (_comparten(k, o[0]), -o[1].area))[2]
    for tabla in tablas:
        if k in tabla:
            return tabla[k]
        cand = [c for c in tabla if _coinciden(k, c)]
        if cand:
            return tabla[max(cand, key=lambda c: (_comparten(k, c), -abs(len(c) - len(k))))]
    return None


def _ubicacion(ciudad: str) -> tuple:
    """(toda la ciudad, barrios oficiales [(clave, polígono, punto interior)], otras tablas clave → punto), una vez
    por ciudad. Las otras tablas, por orden de preferencia: las zonas (comunas, corregimientos…), los lugares de
    OSM dentro de la ciudad y los de alrededor (<ciudad>_lugares.json). Un aviso que dice "Cali" prefiere un
    lugar de Cali."""
    if ciudad not in _UBICACION:
        comunas = cargar("comunas", ciudad)["features"]
        toda = shapely.union_all([shape(f["geometry"]) for f in comunas])
        shapely.prepare(toda)
        oficiales = []
        for f in cargar("barrios", ciudad)["features"]:
            g = shape(f["geometry"])
            oficiales += [(k, g, g.representative_point()) for k in _alias(f["properties"]["nombre"])]
        zonas_, dentro, cerca = {}, {}, {}
        for f in comunas:
            for k in _alias(f["properties"]["nombre"]):
                zonas_.setdefault(k, shape(f["geometry"]).representative_point())
        p = GEO_DIR / f"{slug(ciudad)}_lugares.json"
        for nombre, la, lo in (json.loads(p.read_text(encoding="utf-8"))["lugares"] if p.exists() else []):
            pt = shapely.Point(lo, la)
            for k in _alias(nombre):
                (dentro if toda.contains(pt) else cerca).setdefault(k, pt)
        _UBICACION[ciudad] = toda, oficiales, [zonas_, dentro, cerca]
    return _UBICACION[ciudad]


def corregir_ubicacion(df: pd.DataFrame, solo_imposibles: bool = False) -> pd.DataFrame:
    """Corrige los errores claros en `latitud`/`longitud` (casi siempre de Metrocuadrado, donde el anunciante
    ubica el aviso a mano) y añade `ubicacion`: 'aviso' (la del aviso), 'barrio' (llevada al lugar que nombra)
    o None (sin coordenadas). Lo guardado no cambia: se corrige al leer. Los nombres se comparan con `clave`
    (sin tildes, artículos ni paréntesis, y cada parte de un nombre compuesto: "Tejares - Cristales").
      1. Fuera de Colombia o en (0, 0): sin coordenadas (salían en el mar o en otro país; 132 avisos el 31-oct).
      2. Fuera de su ciudad (de todas sus zonas, con CERCA de margen) y a más de LEJOS_BARRIO km del barrio
         oficial que nombra el aviso con la misma clave (`barrio`, de `ciudad`): al punto interior de ese barrio,
         el más cercano si hay varios. Ejemplo: "Cali, Valle del Lili" ubicado en Palmira.
      3. A más de LEJOS_CIUDAD km de su ciudad: el punto es de otro sitio. Va al lugar que nombra el aviso, con
         la búsqueda más flexible de `_buscar` (un nombre puede contener al otro; barrios oficiales, zonas y
         lugares de OSM: "Cali, Dapa" → Dapa, en Yumbo) y, si no se encuentra, queda sin coordenadas. Ejemplos
         del 1-nov: "Cali, Pance" en Bogotá; "Cali, Nueva Base" en el Meta.
      4. Sin coordenadas (de origen o por la regla 1): al lugar que nombra, como en la 3; si no se encuentra,
         siguen sin ellas. El 1-nov, 3.986 avisos de Metrocuadrado de las 9 ciudades llegaban sin coordenadas
         (Cali 655) y no salían en el mapa; el 86% nombra un lugar que se encuentra.
    "Si se puede, donde toca; si no, fuera del mapa" (pedido del usuario, 1-nov).
    Con `solo_imposibles` (FincaRaíz, cuyas coordenadas sí son fiables: su campo de barrio trae varios nombres y
    la regla 2 movía avisos que de verdad están en Puerto Colombia), solo la 1.
    No se toca lo demás: en otro municipio cercano y sin un barrio de la ciudad con su nombre, el aviso suele
    estar de verdad allí (Villa Campestre, de Puerto Colombia, vendido como Barranquilla); y dentro de la ciudad
    el barrio escrito no es fiable (Bogotá tiene varios «Santa Bárbara»)."""
    df = df.copy()
    lat, lon = pd.to_numeric(df["latitud"], errors="coerce"), pd.to_numeric(df["longitud"], errors="coerce")
    valida = (lat.between(COLOMBIA[0], COLOMBIA[1]) & lon.between(COLOMBIA[2], COLOMBIA[3])).to_numpy()
    lat, lon = np.array(lat.where(valida), dtype=float), np.array(lon.where(valida), dtype=float)   # copias editables
    ubicacion = np.where(valida, "aviso", None).astype(object)
    canon = df["ciudad"].map(canonica).to_numpy()
    nombre = df["barrio"].map(lambda b: clave(b) if isinstance(b, str) else "").to_numpy()
    for ciudad in [] if solo_imposibles else [c for c in CIUDADES if c not in AREAS and archivo("barrios", c).exists()]:
        toda, oficiales, tablas = _ubicacion(ciudad)
        hallado: dict = {}                                                  # muchos avisos repiten el barrio
        buscar = lambda k: hallado[k] if k in hallado else hallado.setdefault(k, _buscar(k, oficiales, tablas))
        for i in np.flatnonzero((canon == ciudad) & ~valida):              # 4: sin coordenadas
            centro = buscar(nombre[i])
            if centro is not None:
                lon[i], lat[i], ubicacion[i] = centro.x, centro.y, "barrio"
        idx = np.flatnonzero((canon == ciudad) & valida)
        if not len(idx):
            continue
        idx = idx[~shapely.dwithin(toda, shapely.points(lon[idx], lat[idx]), CERCA)]      # fuera de la ciudad
        km = shapely.distance(toda, shapely.points(lon[idx], lat[idx])) * 111
        for i, d in zip(idx, km):
            if d > LEJOS_CIUDAD:                                            # 3: el punto es de otro sitio
                centro = buscar(nombre[i])
                if centro is None:
                    lon[i], lat[i], ubicacion[i] = np.nan, np.nan, None
                    continue
            else:                                                           # 2: cerca, pero lejos de su barrio
                p = shapely.Point(lon[i], lat[i])
                opciones = [(g, pt) for c, g, pt in oficiales if c == nombre[i] and nombre[i]]
                if not opciones:
                    continue
                g, centro = min(opciones, key=lambda o: o[0].distance(p))
                if g.distance(p) * 111 <= LEJOS_BARRIO:
                    continue
            lon[i], lat[i], ubicacion[i] = centro.x, centro.y, "barrio"
    df["latitud"], df["longitud"], df["ubicacion"] = lat, lon, ubicacion
    return df


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
