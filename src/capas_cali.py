"""
Capas oficiales de Cali (IDESC, Alcaldía de Santiago de Cali) como variables del modelo de precios (E7).

Mismo servicio WFS que los barrios de src/zonas.py. Se bajan una vez a data/idesc/ (git las ignora; pesan
~48 MB) y se cruzan con cada aviso de Cali por sus coordenadas. Fuera de Cali, todas quedan vacías.

Variables (para cada aviso; todas con el prefijo `cali_`, porque algunas se llaman igual que las de OSM):
  - dist_mio_estacion_km, n_paradas_mio_500m    MIO: estaciones/terminales y paradas (Metrocali)
  - homicidios_500m_3a                            homicidios a 500 m en 2023–2025 (Observatorio de Seguridad)
  - ruido_dia_db, ruido_noche_db                  ruido medido en 2015 (zona que contiene al aviso)
  - inundacion                                    amenaza fluvial o pluvial: 0 ninguna · 1 baja · 2 media ·
                                                  3 alta · 4 no mitigable (POT 2014)
  - barrio_precario                               dentro de un barrio precario (MIHábitat)
  - densidad_arborea                              1 baja … 5 muy alta (DAGMA)
  - dist_colegio_km, dist_universidad_km,         equipamientos del POT
    dist_salud_km, dist_seguridad_km
  - centralidad                                   dentro de una centralidad del POT
  - tratamiento                                   tratamiento urbanístico del POT (consolidación,
                                                  renovación, desarrollo…), como categoría
  - dist_via_arteria_km, dist_via_colectora_km    jerarquía vial del POT

Uso:
    python -m src.capas_cali                  # (re)descarga las capas a data/idesc/
    from src.capas_cali import features_cali
    extra = features_cali(df)                 # SOLO las columnas nuevas (latitud/longitud)
"""
from __future__ import annotations
import json
import time
import urllib.request

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import shape

from src.config import BASE_DIR

DIR = BASE_DIR / "data" / "idesc"
WFS = ("https://ws-idesc.cali.gov.co/geoserver/wfs?service=WFS&version=2.0.0"
       "&request=GetFeature&outputFormat=application/json&srsName=EPSG:4326&typeNames=")
FUENTE = "IDESC – Alcaldía de Santiago de Cali"
CAPAS = {
    "mio_estaciones": "metrocali:sitm_estaciones",
    "mio_paradas": "metrocali:sitm_puntos_parada",
    "homicidios": "seguridad_justicia:ps_os_homicidios",
    "ruido_dia": "expediente_municipal:emc_amb_sos_ruido_semana_diurno",
    "ruido_noche": "expediente_municipal:emc_amb_sos_ruido_semana_nocturno",
    "inundacion_fluvial": "pot_2014:amb_ari_inundacion_fluvial",
    "inundacion_pluvial": "pot_2014:amb_ari_inundacion_pluvial",
    "barrios_precarios": "vivienda:pmih_barrios_precarios",
    "densidad_arborea": "dagma:obs_arb_densidad_arborea",
    "educacion": "pot_2014:eqp_uco_educacion",
    "salud": "pot_2014:eqp_uco_salud",
    "seguridad": "pot_2014:eqp_uba_seguridad_ciudadana",
    "centralidades": "pot_2014:nur_centralidades",
    "tratamientos": "pot_2014:nur_tratamientos_urbanisticos",
    "vias": "pot_2014:mov_jerarquizacion_vial",
}
HOMICIDIOS_DESDE = 2023
RUIDO_MAX_DB = 120            # hay mediciones imposibles (500 dB): se descartan
CALI_LAT = 3.42               # para proyectar a km (equirectangular local; error despreciable en la ciudad)
NIVEL_INUNDACION = [("no mitigable", 4), ("alta", 3), ("media", 2), ("baja", 1), ("torrencial", 3)]
NIVEL_ARBOLES = {"Baja": 1, "Regular": 2, "Media": 3, "Alta": 4, "Muy Alta": 5}


def descargar() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    for nombre, capa in CAPAS.items():
        with urllib.request.urlopen(WFS + capa, timeout=600) as r:
            datos = r.read()
        json.loads(datos)                                   # que sea GeoJSON válido antes de guardarlo
        (DIR / f"{nombre}.geojson").write_bytes(datos)
        print(f"✅ {nombre}: {len(datos) / 1e6:.1f} MB")
        time.sleep(0.5)                                     # cortesía con el servidor de la Alcaldía


def _cargar(nombre: str) -> tuple[list, list[dict]]:
    gj = json.loads((DIR / f"{nombre}.geojson").read_text(encoding="utf-8"))
    feats = [f for f in gj["features"] if f.get("geometry")]
    return [shape(f["geometry"]) for f in feats], [f["properties"] for f in feats]


def _km(geoms) -> np.ndarray:
    """Geometrías lon/lat → km (x hacia el este, y hacia el norte)."""
    escala = np.array([111.32 * np.cos(np.radians(CALI_LAT)), 110.57])
    return shapely.transform(np.asarray(geoms, dtype=object), lambda c: c * escala)


def _dentro(puntos, geoms, valores, defecto=np.nan, agregar=max) -> np.ndarray:
    """Para cada punto, el valor del polígono que lo contiene (si hay varios, `agregar`)."""
    out = np.full(len(puntos), defecto, dtype=object)
    arbol = shapely.STRtree(geoms)
    i_pt, i_pol = arbol.query(puntos, predicate="within")
    vistos: dict[int, list] = {}
    for i, j in zip(i_pt, i_pol):
        if valores[j] is not None and not (isinstance(valores[j], float) and np.isnan(valores[j])):
            vistos.setdefault(i, []).append(valores[j])
    for i, vs in vistos.items():
        out[i] = agregar(vs)
    return out


def _distancia(puntos_km, geoms_km) -> np.ndarray:
    if not len(geoms_km):
        return np.full(len(puntos_km), np.nan)
    arbol = shapely.STRtree(geoms_km)
    _, d = arbol.query_nearest(puntos_km, return_distance=True, all_matches=False)
    return d


def _conteo(puntos_km, geoms_km, radio_km: float) -> np.ndarray:
    arbol = shapely.STRtree(geoms_km)
    i_pt, _ = arbol.query(puntos_km, predicate="dwithin", distance=radio_km)
    return np.bincount(i_pt, minlength=len(puntos_km))


def _nivel_inundacion(tipo: str | None) -> int | None:
    t = (tipo or "").lower()
    return next((n for k, n in NIVEL_INUNDACION if k in t), None)


def features_cali(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de las capas de la IDESC para los avisos de Cali (el resto, vacío)."""
    lat, lon = ("latitud", "longitud") if "latitud" in df else ("Latitud", "Longitud")
    es_cali = (df["ciudad"] if "ciudad" in df else df["Ciudad"]).eq("Cali") & df[lat].notna() & df[lon].notna()
    out = pd.DataFrame(index=df.index)
    if not es_cali.any():
        return out
    d = df[es_cali]
    pts = shapely.points(d[lon].astype(float).to_numpy(), d[lat].astype(float).to_numpy())
    pts_km = _km(pts)
    col = lambda nombre, valores: out.__setitem__(nombre, pd.Series(valores, index=d.index).reindex(df.index))

    # MIO
    g, _ = _cargar("mio_estaciones")
    col("dist_mio_estacion_km", _distancia(pts_km, _km(g)))
    g, p = _cargar("mio_paradas")
    unicas = {}
    for geom, props in zip(g, p):
        unicas.setdefault(props.get("stopid"), geom)
    col("n_paradas_mio_500m", _conteo(pts_km, _km(list(unicas.values())), 0.5))

    # Seguridad: homicidios recientes alrededor
    g, p = _cargar("homicidios")
    recientes = [x for x, props in zip(g, p) if (props.get("homanioh") or 0) >= HOMICIDIOS_DESDE]
    col("homicidios_500m_3a", _conteo(pts_km, _km(recientes), 0.5))

    # Ruido (zona que contiene al aviso)
    for nombre in ("ruido_dia", "ruido_noche"):
        g, p = _cargar(nombre)
        v = [x["laeq_db"] if x.get("laeq_db") and x["laeq_db"] <= RUIDO_MAX_DB else None for x in p]
        col(f"{nombre}_db", _dentro(pts, g, v).astype(float))

    # Amenaza de inundación (la peor de fluvial y pluvial)
    niveles = np.zeros(len(pts))
    for nombre in ("inundacion_fluvial", "inundacion_pluvial"):
        g, p = _cargar(nombre)
        v = [_nivel_inundacion(x.get("tipo_amena")) for x in p]
        niveles = np.fmax(niveles, _dentro(pts, g, v, defecto=0).astype(float))
    col("inundacion", niveles)

    g, _ = _cargar("barrios_precarios")
    col("barrio_precario", _dentro(pts, g, [1] * len(g), defecto=0).astype(float))
    g, p = _cargar("densidad_arborea")
    col("densidad_arborea", _dentro(pts, g, [NIVEL_ARBOLES.get(x.get("texto")) for x in p]).astype(float))

    # Equipamientos (distancia al borde del predio)
    g, p = _cargar("educacion")
    es_u = np.array([(x.get("tipo") or "") == "UNIVERSIDAD" or "SUPERIOR" in (x.get("nivel") or "") for x in p])
    g_km = _km(g)
    col("dist_colegio_km", _distancia(pts_km, g_km[~es_u]))
    col("dist_universidad_km", _distancia(pts_km, g_km[es_u]))
    for nombre in ("salud", "seguridad"):
        g, _ = _cargar(nombre)
        col(f"dist_{nombre}_km", _distancia(pts_km, _km(g)))

    # Norma urbana
    g, _ = _cargar("centralidades")
    col("centralidad", _dentro(pts, g, [1] * len(g), defecto=0).astype(float))
    g, p = _cargar("tratamientos")
    trat = [str(x.get("tratamient") or "").split(" - ")[0].split(" (")[0].strip() or None for x in p]
    col("tratamiento", _dentro(pts, g, trat, agregar=lambda vs: vs[0]))
    out["tratamiento"] = out["tratamiento"].astype("category")

    # Vías
    g, p = _cargar("vias")
    tipo = np.array([x.get("tipo_via") or "" for x in p])
    g_km = _km(g)
    arteria = np.isin(tipo, ["Via Arteria Principal", "Via Arteria Secundaria", "Via Interegional"])
    col("dist_via_arteria_km", _distancia(pts_km, g_km[arteria]))
    col("dist_via_colectora_km", _distancia(pts_km, g_km[tipo == "Via Colectora"]))
    return out.add_prefix("cali_")


if __name__ == "__main__":
    descargar()
