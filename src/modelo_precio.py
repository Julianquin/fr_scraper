"""
Modelo de estimación de precios de vivienda (bloque E7): preparación de datos común a los notebooks
(notebooks/modelo_precio_e7.ipynb, notebooks/comparacion_modelos_e7.ipynb) y, más adelante, al
entrenamiento semanal en producción.

Datos: los avisos de venta ACTIVOS de estado/avisos (src/consolidar.py), solo vivienda
(apartamento, apartaestudio, casa). Son precios pedidos, no de cierre.

Objetivo: log(precio por m²). El precio estimado es exp(predicción) × área.

Validación honesta:
  - por grupos: el mismo inmueble (mismas coordenadas, área y habitaciones, aunque lo publiquen
    anunciantes distintos) nunca queda en entrenamiento y prueba a la vez;
  - temporal: entrenar con lo publicado antes de una fecha y probar con lo publicado después.

Uso (en un notebook):
    from src import modelo_precio as mp
    a, pasos = mp.limpiar(mp.cargar())
    X, TE, amenidades = mp.variables(a)
    y, grupo = a["lpm2"].to_numpy(), mp.grupos(a)
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold

from src.config import BASE_DIR

AVISOS = BASE_DIR / "data" / "estado" / "avisos.parquet"   # actualizar: python -m src.consolidar --bajar
TIPOS = ["Apartamento", "Apartaestudio", "Casa"]
SEMILLA = 0
NUMERICAS = ["m2", "m2_construido", "m2_terreno", "habitaciones", "banos", "parqueaderos", "piso",
             "latitud", "longitud", "n_imagenes"]
CATEGORICAS = ["tipo", "estado", "departamento"]          # pocas categorías: van tal cual
ALTA_CARDINALIDAD = ["ciudad", "barrio", "zona", "comuna"]  # muchas: codificadas por su precio medio
N_AMENIDADES = 40


def cargar(ruta: Path = AVISOS) -> pd.DataFrame:
    """Avisos de vivienda activos, con `es_proyecto`."""
    a = pd.read_parquet(ruta)
    a = a[(a["estado_aviso"] == "activo") & a["tipo"].isin(TIPOS)].copy()
    etiquetas = a["etiquetas"].fillna("[]").map(json.loads)
    a["es_proyecto"] = etiquetas.map(lambda l: "Proyecto" in l or "Unidad de proyecto" in l)
    return a


def limpiar(a: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quita solo errores evidentes (los precios caros o baratos de verdad se quedan).
    Devuelve los avisos limpios con `pm2` y `lpm2`, y una tabla de cuántos quedan en cada paso."""
    pasos = [("Vivienda activa", len(a))]
    a = a[a["precio"].notna() & a["m2"].between(15, 3000)].copy()
    pasos.append(("Con precio en pesos y área 15–3.000 m²", len(a)))
    a["pm2"] = a["precio"] / a["m2"]
    a = a[a["pm2"].between(800_000, 35_000_000)].copy()
    pasos.append(("Precio/m² entre 0,8 y 35 M", len(a)))
    a["lpm2"] = np.log(a["pm2"])
    mediana = a.groupby(["ciudad", "tipo"])["lpm2"].transform("median")
    n_grupo = a.groupby(["ciudad", "tipo"])["lpm2"].transform("size")
    a = a[(n_grupo < 20) | ((a["lpm2"] - mediana).abs() <= 1.2)]       # ~3,3 veces arriba o abajo
    pasos.append(("Sin extremos frente a su ciudad y tipo", len(a)))
    a = a.drop_duplicates(["ciudad", "barrio", "tipo", "precio", "m2", "habitaciones", "banos"]).reset_index(drop=True)
    pasos.append(("Sin duplicados exactos", len(a)))
    return a, pd.DataFrame(pasos, columns=["paso", "avisos"]).set_index("paso")


def amenidades_comunes(a: pd.DataFrame, n: int = N_AMENIDADES) -> list[str]:
    am = a["amenidades"].fillna("").str.split("|").explode()
    return am.value_counts().drop("", errors="ignore").head(n).index.tolist()


def variables(a: pd.DataFrame, amenidades: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(X, TE, amenidades). X: numéricas, categóricas (dtype category) y amenidades 0/1.
    TE: las de alta cardinalidad como texto ('ciudad|barrio'), para codificarlas por objetivo o,
    en modelos que las aceptan (CatBoost), usarlas tal cual."""
    amenidades = amenidades if amenidades is not None else amenidades_comunes(a)
    num = lambda c: pd.to_numeric(a[c], errors="coerce")
    X = pd.DataFrame(index=a.index)
    for c in NUMERICAS:
        X[c] = num(c)
    X["estrato"] = num("estrato").where(num("estrato").between(1, 6))
    X["antiguedad"] = num("antiguedad").where(num("antiguedad").between(1, 5))
    X["admin_m2"] = (num("administracion") / num("m2")).where(num("administracion") > 0)
    X["es_proyecto"] = a["es_proyecto"].astype(int)
    for c in CATEGORICAS:
        X[c] = a[c].fillna("NA").replace("", "NA").astype("category")
    am = a["amenidades"].fillna("")
    for m in amenidades:
        X["am: " + m] = am.str.contains(m, regex=False).astype(int)
    clave = lambda c: (a["ciudad"].fillna("") + "|" + a[c].fillna("")).where(a[c].notna(), "NA")
    TE = pd.DataFrame({"ciudad": a["ciudad"].fillna("NA"), "barrio": clave("barrio"),
                       "zona": clave("zona"), "comuna": clave("comuna")}, index=a.index)
    return X, TE, amenidades


def grupos(a: pd.DataFrame) -> pd.Series:
    """Mismo inmueble = mismas coordenadas (4 decimales), área y habitaciones."""
    return (a["latitud"].round(4).astype(str) + "|" + a["longitud"].round(4).astype(str) + "|"
            + a["m2"].astype(str) + "|" + a["habitaciones"].astype(str))


def particion_grupos(grupo: pd.Series, prueba: float = 0.2, semilla: int = SEMILLA) -> tuple[np.ndarray, np.ndarray]:
    """Índices (posiciones) de entrenamiento y prueba, sin partir un inmueble."""
    return next(GroupShuffleSplit(n_splits=1, test_size=prueba, random_state=semilla)
                .split(np.zeros(len(grupo)), groups=grupo))


def subparticion(indices: np.ndarray, grupo: pd.Series, fraccion: float = 0.1,
                 semilla: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Parte `indices` (posiciones) en ajuste y validación (para parada temprana), por grupos."""
    j_fit, j_val = next(GroupShuffleSplit(n_splits=1, test_size=fraccion, random_state=semilla)
                        .split(indices, groups=grupo.iloc[indices]))
    return indices[j_fit], indices[j_val]


def particion_temporal(a: pd.DataFrame, grupo: pd.Series, corte: str = "2026-10-01") -> tuple[np.ndarray, np.ndarray]:
    """Entrenar con lo publicado antes de `corte`, probar con lo publicado después; se quitan de la
    prueba los inmuebles que ya estaban en el entrenamiento (re-publicaciones)."""
    publicado = a["fecha_publicacion"].astype(str).str[:10]
    tr = np.flatnonzero((publicado < corte).to_numpy())
    te = np.flatnonzero((publicado >= corte).to_numpy() & ~grupo.isin(set(grupo.iloc[tr])).to_numpy())
    return tr, te


def codificador(semilla: int = SEMILLA):
    """TargetEncoder con validación cruzada interna (no filtra la respuesta al entrenar)."""
    from sklearn.preprocessing import TargetEncoder
    return TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=semilla))


def error(pred_log, y_log) -> np.ndarray:
    """Error porcentual absoluto del precio (o del precio/m², que es lo mismo con el área fija)."""
    return np.abs(np.exp(np.asarray(pred_log) - np.asarray(y_log)) - 1)


def metricas(e) -> dict:
    e = np.asarray(e, dtype=float)
    e = e[~np.isnan(e)]
    return {"MdAPE": np.median(e), "±10%": np.mean(e <= .10), "±20%": np.mean(e <= .20), "avisos": len(e)}


NIVELES = [["ciudad", "barrio", "tipo", "estrato"], ["ciudad", "barrio", "tipo"],
           ["ciudad", "tipo", "estrato"], ["ciudad", "tipo"], ["tipo"]]


def linea_base(tr: pd.DataFrame, te: pd.DataFrame, minimo: int = 5) -> np.ndarray:
    """Mediana del log precio/m² de `tr` en el nivel más fino con al menos `minimo` avisos
    (barrio, tipo y estrato → … → tipo). `tr` y `te` necesitan `estrato` ya limpio (1–6)."""
    pred = pd.Series(np.nan, index=te.index)
    for nivel in NIVELES:
        g = tr.groupby(nivel)["lpm2"].agg(["median", "size"])
        g = g.loc[g["size"] >= minimo, "median"]
        m = te[nivel].merge(g, left_on=nivel, right_index=True, how="left")["median"]
        pred = pred.fillna(pd.Series(m.to_numpy(), index=te.index))
    return pred.to_numpy()


# ── Variables nuevas (feature engineering, 27-oct) ─────────────────────────────
# Las que usan el precio (vecinos, puntaje del texto) se calculan FUERA DE PLIEGUE: para los avisos de
# referencia (entrenamiento), con modelos que no los vieron (5 pliegues por inmueble); para el resto,
# con toda la referencia. Las que no usan el precio (servicios, densidad, palabras clave) no filtran nada.

def _pliegues(ref: np.ndarray, grupo: pd.Series, n: int = 5, semilla: int = SEMILLA):
    from sklearn.model_selection import GroupKFold
    codigos = grupo.iloc[ref].astype("category").cat.codes     # con texto, GroupKFold tarda ~1 minuto
    for tr, va in GroupKFold(n_splits=n, shuffle=True, random_state=semilla).split(ref, groups=codigos):
        yield ref[tr], ref[va]


def _km(a: pd.DataFrame) -> np.ndarray:
    """Coordenadas proyectadas a km (equirectangular local): para buscar vecinos dentro de Colombia el
    error frente a la distancia sobre la esfera es despreciable, y un KD-tree es mucho más rápido."""
    lat = a["latitud"].astype(float).to_numpy()
    lon = a["longitud"].astype(float).to_numpy()
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def derivadas(a: pd.DataFrame) -> pd.DataFrame:
    """Proporciones del inmueble: ayudan sobre todo con las casas (lote frente a construcción)."""
    num = lambda c: pd.to_numeric(a[c], errors="coerce")
    hab = num("habitaciones").where(num("habitaciones") > 0)
    return pd.DataFrame({
        "m2_por_hab": num("m2") / hab,
        "banos_por_hab": num("banos") / hab,
        "construido_ratio": num("m2_construido") / num("m2"),
        "terreno_ratio": num("m2_terreno") / num("m2"),
        "n_amenidades": a["amenidades"].fillna("").map(lambda s: len([x for x in s.split("|") if x])),
        "largo_descripcion": a["descripcion"].fillna("").str.len(),
    }, index=a.index)


def contexto(a: pd.DataFrame, radio_km: float = 0.5, k: int = 20) -> pd.DataFrame:
    """Sin usar precios: cuánta oferta hay alrededor y su estrato mediano (útil si el estrato del aviso
    falta o está mal)."""
    from scipy.spatial import cKDTree
    X = _km(a)
    ok = ~np.isnan(X).any(axis=1)
    arbol = cKDTree(X[ok])
    out = pd.DataFrame(index=a.index)
    out.loc[ok, "oferta_500m"] = arbol.query_ball_point(X[ok], r=radio_km, return_length=True, workers=-1) - 1
    _, idx = arbol.query(X[ok], k=k + 1, workers=-1)
    estrato = pd.to_numeric(a["estrato"], errors="coerce").where(lambda s: s.between(1, 6)).to_numpy()[ok]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)        # vecinos sin estrato → vacío
        out.loc[ok, "estrato_alrededor"] = np.nanmedian(estrato[idx[:, 1:]], axis=1)
    out.loc[ok, "apto_share_alrededor"] = (a["tipo"].to_numpy()[ok][idx[:, 1:]] == "Apartamento").mean(axis=1)
    return out


def vecindario(a: pd.DataFrame, y: np.ndarray, ref: np.ndarray, objetivo: np.ndarray, grupo: pd.Series,
               ks: tuple[int, ...] = (10, 30)) -> pd.DataFrame:
    """Los "comparables" como variable: mediana del log precio/m² de los k vecinos más cercanos del
    MISMO TIPO entre los avisos de referencia, y la distancia mediana a ellos (densidad)."""
    from scipy.spatial import cKDTree
    X = _km(a)
    tipo = a["tipo"].to_numpy()
    con_geo = ~np.isnan(X).any(axis=1)
    res = {f"{p}_vecinos_{k}": np.full(len(a), np.nan) for k in ks for p in ("med", "dist")}

    def llenar(base: np.ndarray, consulta: np.ndarray):
        for t in np.unique(tipo[consulta]):
            b = base[(tipo[base] == t) & con_geo[base]]
            q = consulta[(tipo[consulta] == t) & con_geo[consulta]]
            if len(b) < max(ks) or not len(q):
                continue
            dist, idx = cKDTree(X[b]).query(X[q], k=max(ks), workers=-1)
            for k in ks:
                res[f"med_vecinos_{k}"][q] = np.median(y[b][idx[:, :k]], axis=1)
                res[f"dist_vecinos_{k}"][q] = np.median(dist[:, :k], axis=1)

    for tr, va in _pliegues(ref, grupo):
        llenar(tr, va)
    llenar(ref, objetivo)
    return pd.DataFrame(res, index=a.index)


# Palabras de la descripción (sin tildes, minúsculas) → variable 0/1. Solo cosas del INMUEBLE: lo que
# describe al vendedor ("urgente", "remate", "negociable") queda fuera, porque el modelo aprendería que esos
# avisos valen menos y escondería justo las oportunidades que busca la alerta.
CLAVES = {
    "remodelado": r"\bremodelad[oa]s?\b",
    "para_remodelar": r"para remodelar|requiere remodel|necesita remodel|para reformar|para actualizar",
    "obra_gris": r"obra gris|obra negra",
    "para_estrenar": r"para estrenar|a estrenar|sin estrenar",
    "penthouse": r"penthouse|pent house|pent-house",
    "duplex": r"\bduplex\b",
    "terraza": r"\bterraza",
    "vista": r"vista (?:panoramica|a la ciudad|al mar|a la montana|espectacular|hermosa|privilegiada)",
    "esquinero": r"\besquiner[oa]",
    "con_renta": r"con renta|rentando|actualmente arrendad|arrendad[oa] actualmente|con inquilino",
    "sucesion": r"sucesion|herencia",
    "vis": r"\bvis\b|interes social|subsidio",
    "conjunto": r"conjunto cerrado|unidad cerrada|unidad residencial",
    "lote_amplio": r"lote (?:amplio|grande|de \d)|amplio lote",
}


def _normalizar(s: pd.Series) -> pd.Series:
    import unicodedata
    return s.fillna("").map(lambda t: unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower())


def texto_claves(a: pd.DataFrame) -> pd.DataFrame:
    t = _normalizar(a["descripcion"])
    return pd.DataFrame({f"txt_{k}": t.str.contains(p, regex=True).astype(int) for k, p in CLAVES.items()},
                        index=a.index)


# Fuera del vocabulario del puntaje: palabras sobre el precio o la urgencia del vendedor (misma razón que
# en CLAVES). Los NÚMEROS también se quitan: muchas descripciones traen el precio ("350.000.000") y el
# modelo aprendería a leerlo en vez de estimarlo.
PALABRAS_VENDEDOR = ["precio", "precios", "valor", "millones", "millon", "negociable", "negociables", "oportunidad",
                     "remate", "urgente", "urge", "ganga", "barato", "barata", "economico", "economica", "rebajado",
                     "rebaja", "descuento", "oferta", "liquidacion", "ocasion", "pesos", "cop", "mill", "mm"]


def texto_puntaje(a: pd.DataFrame, y: np.ndarray, ref: np.ndarray, objetivo: np.ndarray,
                  grupo: pd.Series) -> pd.Series:
    """Lo que dice TODA la descripción sobre el inmueble: un modelo lineal sobre palabras y pares de
    palabras (TF-IDF, sin números ni PALABRAS_VENDEDOR) predice el log precio/m²; esa predicción entra
    como una variable más."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    t = _normalizar(a["descripcion"]).str.replace(r"\d", " ", regex=True).to_numpy()
    out = pd.Series(np.nan, index=a.index, name="txt_puntaje")

    def ajustar(base: np.ndarray, consulta: np.ndarray):
        vec = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 2), sublinear_tf=True, max_features=200_000,
                              token_pattern=r"(?u)\b[a-z]{2,}\b", stop_words=PALABRAS_VENDEDOR)
        m = Ridge(alpha=3.0).fit(vec.fit_transform(t[base]), y[base])
        out.iloc[consulta] = m.predict(vec.transform(t[consulta]))

    for tr, va in _pliegues(ref, grupo):
        ajustar(tr, va)
    ajustar(ref, objetivo)
    return out


def servicios(a: pd.DataFrame) -> pd.DataFrame:
    """Distancias y conteos de servicios de OpenStreetMap (src/osm_pois.py)."""
    from src.osm_pois import features_servicios
    return features_servicios(a)


# ── Configuración elegida y entrenamiento (variables_e7.ipynb) ────────────────
LGBM = dict(n_estimators=10000, learning_rate=0.03, num_leaves=127, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, random_state=SEMILLA, verbose=-1)


def variables_nuevas(a: pd.DataFrame, y: np.ndarray, ref: np.ndarray, objetivo: np.ndarray, grupo: pd.Series,
                     grupos: tuple[str, ...] = ("mercado", "servicios", "texto")) -> dict[str, pd.DataFrame]:
    """Los grupos de variables nuevas (las que usan precios, fuera de pliegue con `ref` como referencia).
    Por defecto, los tres que entraron en la ablación; "censo" (DANE) y "cali" (IDESC) son opcionales."""
    constructores = {
        "mercado": lambda: pd.concat([vecindario(a, y, ref, objetivo, grupo), contexto(a)], axis=1),
        "servicios": lambda: servicios(a),
        "texto": lambda: pd.concat([texto_claves(a), texto_puntaje(a, y, ref, objetivo, grupo)], axis=1),
        "censo": lambda: __import__("src.censo_dane", fromlist=["x"]).features_censo(a),
        "cali": lambda: __import__("src.capas_cali", fromlist=["x"]).features_cali(a),
        "derivadas": lambda: derivadas(a),
    }
    return {g: constructores[g]() for g in grupos}


def con_te(Xb: pd.DataFrame, TE: pd.DataFrame, y: np.ndarray, ajuste: np.ndarray, *otros) -> list[pd.DataFrame]:
    """Xb + barrio/zona/comuna/ciudad codificados por su precio medio (ajustado SOLO con `ajuste`);
    nombres de columna sin ':' ni espacios (LightGBM no los acepta)."""
    enc = codificador()
    Xa = Xb.iloc[ajuste].copy()
    Xa[["te_" + c for c in TE.columns]] = enc.fit_transform(TE.iloc[ajuste], y[ajuste])
    salida = [Xa]
    for idx in otros:
        Xo = Xb.iloc[idx].copy()
        Xo[["te_" + c for c in TE.columns]] = enc.transform(TE.iloc[idx])
        salida.append(Xo)
    return [Z.rename(columns=lambda c: c.replace(":", "_").replace(" ", "_")) for Z in salida]


def lightgbm(Xb: pd.DataFrame, TE: pd.DataFrame, y: np.ndarray, ajuste: np.ndarray, validacion: np.ndarray,
             prueba: np.ndarray, devolver_modelo: bool = False):
    """Entrena con `ajuste` (parada temprana con `validacion`) y predice el log precio/m² de `prueba`."""
    import lightgbm as lgb
    Xa, Xv, Xp = con_te(Xb, TE, y, ajuste, validacion, prueba)
    m = lgb.LGBMRegressor(**LGBM)
    m.fit(Xa, y[ajuste], eval_X=Xv, eval_y=y[validacion], callbacks=[lgb.early_stopping(200, verbose=False)])
    return (m.predict(Xp), m, Xp) if devolver_modelo else m.predict(Xp)
