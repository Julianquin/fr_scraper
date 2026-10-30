"""
Polígonos oficiales de las ciudades principales (bloque E6), con el mismo formato que los de Cali
(src/zonas.py) para que el mapa, la alerta y el informe funcionen en cualquiera de ellas.

Para cada ciudad deja en data/geo/ (se versionan: el CI no depende de que las fuentes estén arriba):
  <slug>_comunas.geojson   la división principal (comunas o localidades): codigo, nombre y etiqueta (el número
                           oficial para el mapa, o vacía si no lo tiene)
  <slug>_barrios.geojson   los barrios, si hay capa: codigo, nombre, comuna (codigo de su zona)

Fuentes (31-oct-2026):
  Bogotá        localidades: Secretaría Distrital de Planeación · barrios: sectores catastrales de la UAECD
                (Datos Abiertos Bogotá, CC BY 4.0)
  Medellín      comunas y corregimientos, barrios y veredas: Subsecretaría de Catastro (servidor de mapas
                de la Alcaldía)
  Barranquilla  barrios del POT 2024 (datos.gov.co); sus 5 localidades, de OSM (en el POT hay barrios con dos
                localidades o ninguna): cada barrio va a la que contiene su punto interior
  Bucaramanga, Manizales, Armenia, Pereira
                comunas de OpenStreetMap (límites administrativos de nivel 8 dentro del municipio, del extract
                de Colombia en data/osm/, ODbL)
Cali tiene las suyas de la IDESC (src/zonas.py). Cartagena no tiene capa oficial publicada: va sin mapa.

Lo descargado se guarda en data/geo_fuentes/ (local, no se versiona) y se reutiliza: algunos servidores cortan
descargas largas, y así basta con bajarlo una vez.

Uso:
    python -m src.geo_ciudades                 # todas
    python -m src.geo_ciudades --ciudad Bogotá
"""
from __future__ import annotations
import argparse
import glob
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile

import pandas as pd
import shapely
from shapely.geometry import mapping, shape
from shapely.ops import polygonize, unary_union

from src.config import BASE_DIR
from src.zonas import GEO_DIR, TOLERANCIA, slug

MEDELLIN = "https://www.medellin.gov.co/servidormapas/rest/services/ServiciosCatastro/Base_Catastral/MapServer"
FUENTES = {
    "bogota_localidades": "https://datosabiertos.bogota.gov.co/dataset/856cb657-8ca3-4ee8-857f-37211173b1f8/"
                          "resource/497b8756-0927-4aee-8da9-ca4e32ca3a8a/download/loca.json",
    "bogota_sectores": "https://datosabiertos.bogota.gov.co/dataset/6f5031cf-07d0-42c3-ba5c-4436eba0a2d9/"
                       "resource/428a5f4e-b9d4-4d91-88a7-28fb404a08ab/download/sector.geojson.09.26.zip",
    "medellin_comunas": f"{MEDELLIN}/0",
    "medellin_barrios": f"{MEDELLIN}/1",
    "barranquilla_barrios": "https://www.datos.gov.co/resource/qui6-qeux.geojson?$limit=5000",
}
CACHE = BASE_DIR / "data" / "geo_fuentes"
COPIA = {"bogota_localidades": "bogota_localidades_oficial.json", "bogota_sectores": "bogota_sectores_catastrales.geojson",
         "medellin_comunas": "medellin_comunas.geojson", "medellin_barrios": "medellin_barrios.geojson",
         "barranquilla_barrios": "barranquilla_barrios_pot.geojson"}
OSM_MUNICIPIOS = {"Barranquilla": 1335179, "Bucaramanga": 1423498, "Manizales": 385842, "Armenia": 1343039,
                  "Pereira": 1343043}
SOLO_OSM = ["Bucaramanga", "Manizales", "Armenia", "Pereira"]       # comunas de OSM y sin barrios


def _json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=300) as r:
        return json.load(r)


def _fuente(nombre: str) -> dict:
    """La fuente `nombre`: la copia de data/geo_fuentes/ si está; si no, la descarga y la guarda."""
    copia = CACHE / COPIA[nombre]
    if copia.exists():
        return json.loads(copia.read_text(encoding="utf-8"))
    url = FUENTES[nombre]
    if nombre.startswith("medellin"):
        datos = {"type": "FeatureCollection", "features": _arcgis(url)}
    elif url.endswith(".zip"):
        with urllib.request.urlopen(url, timeout=300) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        datos = json.loads(z.read([n for n in z.namelist() if n.lower().endswith("json")][0]))
    else:
        datos = _json(url)
    CACHE.mkdir(parents=True, exist_ok=True)
    copia.write_text(json.dumps(datos, ensure_ascii=False), encoding="utf-8")
    return datos


def _arcgis(capa: str) -> list[dict]:
    """Todas las entidades de una capa de ArcGIS Server, en WGS84 (de 500 en 500)."""
    features, desde = [], 0
    while True:
        q = urllib.parse.urlencode({"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                                    "resultOffset": desde, "resultRecordCount": 500})
        nuevas = _json(f"{capa}/query?{q}").get("features", [])
        features += nuevas
        if len(nuevas) < 500:
            return features
        desde += 500


# Tildes y números romanos que las fuentes no traen (Catastro escribe en mayúsculas sin tildes)
ARREGLOS = {"usaquen": "Usaquén", "martires": "Mártires", "fontibon": "Fontibón", "engativa": "Engativá",
            "bolivar": "Bolívar", "cristobal": "Cristóbal", "belen": "Belén", "america": "América", "area": "Área",
            "expansion": "Expansión", "rincon": "Rincón", "nicolas": "Nicolás", "jose": "José", "maria": "María",
            "ii": "II", "iii": "III", "iv": "IV"}


def _titulo(s: str) -> str:
    """'ANTONIO NARIÑO' → 'Antonio Nariño', 'USAQUEN' → 'Usaquén' (respeta las palabras cortas)."""
    menores = {"de", "del", "la", "las", "los", "el", "y"}
    palabras = str(s).strip().lower().split()
    return " ".join(p if i and p in menores else ARREGLOS.get(p, p[:1].upper() + p[1:]) for i, p in enumerate(palabras))


def _con_comuna(barrios: list[tuple[dict, object]], comunas: list[tuple[dict, object]]) -> None:
    """Pone a cada barrio el codigo de la comuna que contiene su punto interior."""
    arbol = shapely.STRtree([g for _, g in comunas])
    for props, g in barrios:
        i = arbol.query(g.representative_point(), predicate="within")
        props["comuna"] = comunas[i[0]][0]["codigo"] if len(i) else None


def _guardar(nombre: str, feats: list[tuple[dict, object]], fuente: str) -> None:
    salida = []
    for props, g in feats:
        g = shapely.make_valid(g)
        geom = shapely.set_precision(g.simplify(TOLERANCIA), 1e-5)
        if not geom.is_empty:
            salida.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
    p = GEO_DIR / f"{nombre}.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "fuente": fuente, "features": salida},
                            ensure_ascii=False), encoding="utf-8")
    print(f"✅ {len(salida)} polígonos → {p.relative_to(BASE_DIR)} ({p.stat().st_size / 1e3:.0f} KB)")


def bogota() -> None:
    loc = _fuente("bogota_localidades")                       # Esri JSON (EPSG 4686 ≈ WGS84)
    comunas = [({"codigo": f["attributes"]["LocCodigo"].zfill(2), "nombre": _titulo(f["attributes"]["LocNombre"]),
                 "etiqueta": str(int(f["attributes"]["LocCodigo"]))},
                unary_union([shapely.Polygon(r) for r in f["geometry"]["rings"]]))
               for f in loc["features"]]
    sectores = _fuente("bogota_sectores")
    barrios = [({"codigo": f["properties"]["SCACODIGO"], "nombre": _titulo(f["properties"]["SCANOMBRE"])},
                shape(f["geometry"])) for f in sectores["features"] if f.get("geometry")]
    _con_comuna(barrios, comunas)
    fuente = "Datos Abiertos Bogotá: localidades (SDP) y sectores catastrales (UAECD), CC BY 4.0"
    _guardar("bogota_comunas", comunas, fuente)
    _guardar("bogota_barrios", barrios, fuente)


def medellin() -> None:
    comunas = [({"codigo": str(f["properties"]["comuna"]).zfill(2), "nombre": _titulo(f["properties"]["nombre"]),
                 "etiqueta": str(int(f["properties"]["comuna"]))},
                shape(f["geometry"])) for f in _fuente("medellin_comunas")["features"] if f.get("geometry")]
    barrios = [({"codigo": str(f["properties"]["codigo"]), "nombre": _titulo(f["properties"]["nombre_barrio"]),
                 "comuna": str(f["properties"]["comuna"]).zfill(2)}, shape(f["geometry"]))
               for f in _fuente("medellin_barrios")["features"] if f.get("geometry")]
    fuente = "Alcaldía de Medellín, Subsecretaría de Catastro"
    _guardar("medellin_comunas", comunas, fuente)
    _guardar("medellin_barrios", barrios, fuente)


def barranquilla(rels: pd.DataFrame) -> None:
    comunas = [({"codigo": c["codigo"], "nombre": re.sub(r"^Localidad\s+", "", c["nombre"]), "etiqueta": ""}, g)
               for c, g in _comunas_osm(rels, "Barranquilla")]
    barrios = [({"codigo": str(f["properties"]["id"]), "nombre": _titulo(f["properties"]["nombre"])}, shape(f["geometry"]))
               for f in _fuente("barranquilla_barrios")["features"] if f.get("geometry")]
    _con_comuna(barrios, comunas)
    fuente = "Alcaldía de Barranquilla, barrios del POT 2024 (datos.gov.co); localidades de OpenStreetMap (ODbL)"
    _guardar("barranquilla_comunas", comunas, fuente)
    _guardar("barranquilla_barrios", barrios, fuente)


def _osm_limites() -> pd.DataFrame:
    """Polígonos de los límites administrativos de OSM (niveles 6 a 10) del extract de Colombia."""
    import duckdb
    pbf = sorted(glob.glob(str(BASE_DIR / "data" / "osm" / "*.pbf")))[-1]
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute(f"""CREATE TEMP TABLE r AS SELECT id, tags, refs, ref_types, ref_roles FROM ST_ReadOSM('{pbf}')
                   WHERE kind = 'relation' AND tags['boundary'] = 'administrative'
                     AND tags['admin_level'] IN ('6','7','8','9','10')""")
    con.execute("""CREATE TEMP TABLE m AS SELECT id AS rel, unnest(refs) AS way, unnest(ref_types) AS tipo,
                   unnest(ref_roles) AS rol FROM r""")
    con.execute("DELETE FROM m WHERE tipo <> 'way' OR rol NOT IN ('outer', 'inner', '')")
    con.execute(f"""CREATE TEMP TABLE w AS SELECT o.id, o.refs FROM ST_ReadOSM('{pbf}') o
                   WHERE o.kind = 'way' AND o.id IN (SELECT DISTINCT way FROM m)""")
    con.execute("CREATE TEMP TABLE wn AS SELECT id, unnest(refs) AS nid, generate_subscripts(refs, 1) AS pos FROM w")
    con.execute(f"""CREATE TEMP TABLE n AS SELECT o.id, o.lat, o.lon FROM ST_ReadOSM('{pbf}') o
                   WHERE o.kind = 'node' AND o.id IN (SELECT DISTINCT nid FROM wn)""")
    lineas = con.execute("""SELECT wn.id, list(n.lon ORDER BY wn.pos) AS xs, list(n.lat ORDER BY wn.pos) AS ys
                            FROM wn JOIN n ON n.id = wn.nid GROUP BY wn.id""").df().set_index("id")
    miembros = con.execute("SELECT rel, way, rol FROM m").df()
    rels = con.execute("SELECT id, tags['admin_level'] AS nivel, tags['name'] AS nombre FROM r").df()
    traza = lambda w: shapely.LineString(list(zip(lineas.at[w, "xs"], lineas.at[w, "ys"])))
    geoms = {}
    for rel, g in miembros.groupby("rel"):
        ways = [w for w in g["way"] if w in lineas.index and len(lineas.at[w, "xs"]) >= 2]
        polys = list(polygonize(unary_union([traza(w) for w in ways]))) if ways else []
        if polys:
            huecos = [shapely.Polygon(traza(w).coords) for w in g.loc[g["rol"] == "inner", "way"]
                      if w in ways and traza(w).is_closed and len(lineas.at[w, "xs"]) >= 4]
            geoms[rel] = unary_union(polys).difference(unary_union(huecos)) if huecos else unary_union(polys)
    rels["geometry"] = rels["id"].map(geoms)
    return rels.dropna(subset=["geometry"])


def _comunas_osm(rels: pd.DataFrame, ciudad: str) -> list[tuple[dict, object]]:
    """Las divisiones de nivel 8 de OSM dentro del municipio: «Comuna 12 - …» se queda con su número."""
    sub = rels[rels["nivel"] == "8"]
    muni = rels.loc[rels["id"] == OSM_MUNICIPIOS[ciudad], "geometry"].iloc[0]
    dentro = sub[shapely.contains(muni, shapely.point_on_surface(sub["geometry"].to_numpy()))]
    comunas = []
    for i, (_, r) in enumerate(dentro.sort_values("nombre").iterrows(), 1):
        m = re.match(r"comuna\s+(\d+)", str(r["nombre"]), re.I)
        comunas.append(({"codigo": m.group(1).zfill(2) if m else f"{i + 50:02d}", "nombre": str(r["nombre"]),
                         "etiqueta": m.group(1) if m else ""},
                        r["geometry"]))
    return comunas


def osm(rels: pd.DataFrame, ciudad: str) -> None:
    _guardar(f"{slug(ciudad)}_comunas", _comunas_osm(rels, ciudad),
             "OpenStreetMap (ODbL), límites administrativos de nivel 8")


CONSTRUCTORES = {"Bogotá": bogota, "Medellín": medellin}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Polígonos oficiales de las ciudades principales (E6)")
    p.add_argument("--ciudad", help="solo esta (por defecto, todas)")
    a = p.parse_args()
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    for ciudad, f in CONSTRUCTORES.items():
        if not a.ciudad or a.ciudad == ciudad:
            f()
    con_osm = [c for c in ["Barranquilla", *SOLO_OSM] if not a.ciudad or a.ciudad == c]
    if con_osm:
        rels = _osm_limites()
        for ciudad in con_osm:
            barranquilla(rels) if ciudad == "Barranquilla" else osm(rels, ciudad)
