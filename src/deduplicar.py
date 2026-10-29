"""
Deduplicación entre portales (bloque E8): qué avisos de Metrocuadrado y de FincaRaíz son el mismo inmueble.

Ningún campo coincide exacto entre portales (cada uno geocodifica a su manera, el área se redondea distinto,
el precio puede diferir), así que se combinan señales para cada par candidato:
  - distancia entre las coordenadas,
  - diferencia de área (la mejor combinación de área total y construida: cada portal publica una distinta)
    y de precio (en log: simétrica); precio idéntico suma,
  - habitaciones, baños y estrato iguales (habitaciones y baños topados en 5: Metrocuadrado publica 5 como
    «5 o más»); PISO distinto resta fuerte (revisión del 27-oct: la misma inmobiliaria usa el mismo texto
    para varios apartamentos de un edificio, y solo el piso los distingue), y habitaciones Y baños distintos
    a la vez también restan (revisión del 29-oct: casas iguales en área y precio pero con otra ficha),
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
TOPE = 5            # habitaciones y baños: Metrocuadrado publica 5 como «5 o más»


def _km(d: pd.DataFrame) -> np.ndarray:
    lat = pd.to_numeric(d["latitud"], errors="coerce").to_numpy()
    lon = pd.to_numeric(d["longitud"], errors="coerce").to_numpy()
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def _texto(d: pd.DataFrame) -> pd.Series:
    import unicodedata
    t = d["descripcion"].fillna("").map(lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())
    return t.str.replace(r"\s+", " ", regex=True).str.strip()


def _num(d: pd.DataFrame, c: str) -> np.ndarray:
    return pd.to_numeric(d[c], errors="coerce").where(lambda v: v > 0).to_numpy()   # 0 = sin dato


def diferencia_area(a: pd.DataFrame, b: pd.DataFrame, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """|log| de la razón de áreas con la mejor combinación de área total y construida: cada portal
    publica una distinta. NaN si ninguno de los dos tiene área."""
    areas = [np.abs(np.log(_num(a, ca)[i] / _num(b, cb)[j])) for ca in ("m2", "m2_construido") for cb in ("m2", "m2_construido")]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # pares sin ninguna área: NaN
        return np.nanmin(np.vstack(areas), axis=0)


def matriz_texto(textos: pd.Series):
    """TF-IDF de fragmentos de caracteres (3–5) con filas normalizadas: el producto de dos filas es el
    coseno entre dos descripciones. Con hashing, sin vocabulario, para que alcance la memoria con todo el
    país (~270.000 descripciones); da los mismos pares que el TF-IDF con vocabulario (Cali, 29-oct)."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 20, alternate_sign=False,
                           norm=None, dtype=np.float32)
    return TfidfTransformer(sublinear_tf=True).fit_transform(hv.transform(textos)).astype(np.float32)


def similitud(Ta, Tb, i: np.ndarray, j: np.ndarray, lote: int = 100_000) -> np.ndarray:
    """Coseno entre la fila `i` de Ta y la `j` de Tb, par a par y por tandas."""
    sim = np.zeros(len(i), dtype=np.float32)
    for k in range(0, len(i), lote):
        s = slice(k, k + lote)
        sim[s] = np.asarray(Ta[i[s]].multiply(Tb[j[s]]).sum(axis=1)).ravel()
    return sim


def señales(a: pd.DataFrame, b: pd.DataFrame, i: np.ndarray, j: np.ndarray, Ta=None, Tb=None) -> pd.DataFrame:
    """Las señales de cada par (fila `i` de `a`, fila `j` de `b`; `a` y `b` pueden ser la misma tabla).
    Ta/Tb: matrices de texto ya calculadas (matriz_texto), para no repetirlas."""
    ta, tb = _texto(a), _texto(b)
    if Ta is None:
        Ta = Tb = matriz_texto(ta) if a is b else None
        if Ta is None:
            T = matriz_texto(pd.concat([ta, tb], ignore_index=True))
            Ta, Tb = T[:len(a)], T[len(a):]
    sim = similitud(Ta, Tb, i, j)
    sim[(ta.str.len().to_numpy()[i] < 30) | (tb.str.len().to_numpy()[j] < 30)] = np.nan   # sin texto útil
    def igual(c: str, tope: float = np.inf) -> np.ndarray:
        va, vb = np.minimum(_num(a, c)[i], tope), np.minimum(_num(b, c)[j], tope)
        return np.where(np.isnan(va) | np.isnan(vb), np.nan, (va == vb).astype(float))
    pa, pb = _num(a, "piso")[i], _num(b, "piso")[j]
    with np.errstate(all="ignore"):
        return pd.DataFrame({
            "i": i, "j": j,
            "dist_m": np.hypot(*(_km(a)[i] - _km(b)[j]).T) * 1000,
            "d_area": diferencia_area(a, b, i, j),
            "piso_distinto": np.where(np.isnan(pa) | np.isnan(pb), np.nan, (pa != pb).astype(float)),
            "d_precio": np.abs(np.log(_num(a, "precio")[i] / _num(b, "precio")[j])),
            "hab_igual": igual("habitaciones", TOPE),
            "banos_igual": igual("banos", TOPE),
            "estrato_igual": igual("estrato"),
            "sim_texto": sim,
        })


def pares(a: pd.DataFrame, b: pd.DataFrame, radio_km: float = RADIO_KM) -> pd.DataFrame:
    """Pares candidatos (mismo tipo, a menos de `radio_km`) entre los avisos de `a` y `b`, con sus señales.
    Índices `i` (posición en `a`) y `j` (posición en `b`)."""
    from scipy.spatial import cKDTree
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    Xa, Xb = _km(a), _km(b)
    ok_a, ok_b = ~np.isnan(Xa).any(axis=1), ~np.isnan(Xb).any(axis=1)
    arbol = cKDTree(Xb[ok_b])
    idx_b = np.flatnonzero(ok_b)
    vecinos = arbol.query_ball_point(Xa[ok_a], r=radio_km, workers=-1)
    i = np.repeat(np.flatnonzero(ok_a), [len(v) for v in vecinos])
    j = idx_b[np.concatenate([np.asarray(v, dtype=int) for v in vecinos])] if len(i) else np.array([], dtype=int)
    mismo_tipo = a["tipo"].to_numpy()[i] == b["tipo"].to_numpy()[j]
    return señales(a, b, i[mismo_tipo], j[mismo_tipo])


def puntaje(P: pd.DataFrame) -> pd.Series:
    """Qué tan probable es que el par sea el mismo inmueble (0 a ~1). Reglas simples y legibles: el texto
    pesa más; área, habitaciones y baños tienen que cuadrar; la distancia y el precio suman. Piso distinto
    casi veta; habitaciones y baños distintos a la vez restan, pero un texto casi idéntico aún los junta."""
    texto = P["sim_texto"].fillna(0)
    ficha = ((P["d_area"] <= 0.05).astype(float) * 0.5 + P["hab_igual"].fillna(0.5) * 0.25 + P["banos_igual"].fillna(0.5) * 0.25)
    cerca = np.clip(1 - P["dist_m"] / 300, 0, 1)
    precio = np.clip(1 - P["d_precio"].fillna(1) / 0.15, 0, 1) + 0.5 * (P["d_precio"] < 0.005)   # idéntico suma
    base = 0.45 * texto + 0.30 * ficha + 0.15 * cerca + 0.10 * precio
    otra_ficha = (P["hab_igual"] == 0) & (P["banos_igual"] == 0)
    return base * np.where(P["piso_distinto"] == 1, 0.4, 1.0) * np.where(otra_ficha, 0.8, 1.0)


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
