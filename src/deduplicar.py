"""
Deduplicación entre portales (bloque E8): qué avisos de Metrocuadrado y de FincaRaíz son el mismo inmueble.

Ningún campo coincide exacto entre portales (cada uno geocodifica a su manera, el área se redondea distinto,
el precio puede diferir), así que se combinan señales para cada par candidato:
  - distancia entre las coordenadas,
  - diferencia de área (la mejor combinación de área total y construida: cada portal publica una distinta)
    y de precio (en log: simétrica); precio idéntico suma,
  - habitaciones, baños y estrato iguales; PISO distinto resta fuerte (revisión del 27-oct: la misma
    inmobiliaria usa el mismo texto para varios apartamentos de un edificio, y solo el piso los distingue),
  - parecido de la descripción: coseno TF-IDF de fragmentos de caracteres. Las inmobiliarias suelen copiar el
    mismo texto en los dos portales, así que es la señal más fuerte.

Candidatos: mismo tipo a menos de RADIO_KM. Luego un emparejamiento uno a uno: el par más parecido primero,
y cada aviso queda en un solo par.

Uso:
    from src.deduplicar import pares, emparejar
    P = pares(m2, fr)                 # todas las señales de cada par candidato
    E = emparejar(P, umbral=…)        # pares aceptados, uno a uno
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd

RADIO_KM = 0.5


def _km(d: pd.DataFrame) -> np.ndarray:
    lat = pd.to_numeric(d["latitud"], errors="coerce").to_numpy()
    lon = pd.to_numeric(d["longitud"], errors="coerce").to_numpy()
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def _texto(d: pd.DataFrame) -> pd.Series:
    import unicodedata
    t = d["descripcion"].fillna("").map(lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())
    return t.str.replace(r"\s+", " ", regex=True).str.strip()


def pares(a: pd.DataFrame, b: pd.DataFrame, radio_km: float = RADIO_KM) -> pd.DataFrame:
    """Pares candidatos (mismo tipo, a menos de `radio_km`) entre los avisos de `a` y `b`, con sus señales.
    Índices `i` (posición en `a`) y `j` (posición en `b`)."""
    from scipy.spatial import cKDTree
    from sklearn.feature_extraction.text import TfidfVectorizer
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    Xa, Xb = _km(a), _km(b)
    ok_a, ok_b = ~np.isnan(Xa).any(axis=1), ~np.isnan(Xb).any(axis=1)
    arbol = cKDTree(Xb[ok_b])
    idx_b = np.flatnonzero(ok_b)
    vecinos = arbol.query_ball_point(Xa[ok_a], r=radio_km, workers=-1)
    i = np.repeat(np.flatnonzero(ok_a), [len(v) for v in vecinos])
    j = idx_b[np.concatenate([np.asarray(v, dtype=int) for v in vecinos])] if len(i) else np.array([], dtype=int)
    mismo_tipo = a["tipo"].to_numpy()[i] == b["tipo"].to_numpy()[j]
    i, j = i[mismo_tipo], j[mismo_tipo]

    num = lambda d, c: pd.to_numeric(d[c], errors="coerce").where(lambda v: v > 0).to_numpy()   # 0 = sin dato
    igual = lambda c: np.where(np.isnan(num(a, c)[i]) | np.isnan(num(b, c)[j]), np.nan,
                               (num(a, c)[i] == num(b, c)[j]).astype(float))
    ta, tb = _texto(a), _texto(b)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    vec.fit(pd.concat([ta, tb]))
    Ta, Tb = vec.transform(ta), vec.transform(tb)
    sim = np.zeros(len(i))
    for k in range(0, len(i), 200_000):                       # coseno par a par, por tandas
        s = slice(k, k + 200_000)
        sim[s] = np.asarray(Ta[i[s]].multiply(Tb[j[s]]).sum(axis=1)).ravel()
    sim[(ta.str.len().to_numpy()[i] < 30) | (tb.str.len().to_numpy()[j] < 30)] = np.nan   # sin texto útil

    areas = [np.abs(np.log(num(a, ca)[i] / num(b, cb)[j])) for ca in ("m2", "m2_construido") for cb in ("m2", "m2_construido")]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # pares sin ninguna área: NaN
        d_area = np.nanmin(np.vstack(areas), axis=0)
    pa, pb = num(a, "piso")[i], num(b, "piso")[j]
    return pd.DataFrame({
        "i": i, "j": j,
        "dist_m": np.hypot(*(Xa[i] - Xb[j]).T) * 1000,
        "d_area": d_area,
        "piso_distinto": np.where(np.isnan(pa) | np.isnan(pb), np.nan, (pa != pb).astype(float)),
        "d_precio": np.abs(np.log(num(a, "precio")[i] / num(b, "precio")[j])),
        "hab_igual": igual("habitaciones"),
        "banos_igual": igual("banos"),
        "estrato_igual": igual("estrato"),
        "sim_texto": sim,
    })


def puntaje(P: pd.DataFrame) -> pd.Series:
    """Qué tan probable es que el par sea el mismo inmueble (0 a ~1). Reglas simples y legibles: el texto
    pesa más; área, habitaciones y baños tienen que cuadrar; la distancia y el precio suman."""
    texto = P["sim_texto"].fillna(0)
    ficha = ((P["d_area"] <= 0.05).astype(float) * 0.5 + P["hab_igual"].fillna(0.5) * 0.25 + P["banos_igual"].fillna(0.5) * 0.25)
    cerca = np.clip(1 - P["dist_m"] / 300, 0, 1)
    precio = np.clip(1 - P["d_precio"].fillna(1) / 0.15, 0, 1) + 0.5 * (P["d_precio"] < 0.005)   # idéntico suma
    base = 0.45 * texto + 0.30 * ficha + 0.15 * cerca + 0.10 * precio
    return base * np.where(P["piso_distinto"] == 1, 0.4, 1.0)


def emparejar(P: pd.DataFrame, umbral: float = 0.6) -> pd.DataFrame:
    """Uno a uno: el par con más puntaje primero; cada aviso queda en un solo par."""
    P = P.assign(puntaje=puntaje(P)).sort_values("puntaje", ascending=False)
    P = P[P["puntaje"] >= umbral]
    usados_i, usados_j, filas = set(), set(), []
    for r in P.itertuples():
        if r.i in usados_i or r.j in usados_j:
            continue
        usados_i.add(r.i); usados_j.add(r.j)
        filas.append(r.Index)
    return P.loc[filas]
