"""
Censo 2018 (DANE) por manzana como variables del modelo de precios (E7): para todo el país.

Fuentes (públicas, geoportal del DANE; se bajan a data/dane/, que git ignora):
  - MGN2018 integrado con el CNPV2018, nivel manzana (SHP_MGN2018_INTGRD_MANZ.zip, ~320 MB):
    505.000 manzanas con estrato (según la factura de energía), tipo y ocupación de las viviendas,
    servicios, personas y nivel educativo.
  - Vulnerabilidad e IPM por manzana (SHP_VULNRB_IPMxMZ.zip, ~250 MB): índice de pobreza multidimensional.
  https://www.dane.gov.co/files/geoportal-provisional/

`preparar()` reduce los shapefiles a un parquet liviano (centroide + conteos) en data/dane/. Para cada aviso
se SUMAN las manzanas cuyo centroide está a menos de RADIO_M (un punto puede caer en la calle y una sola
manzana puede tener muy pocos datos) y se calculan proporciones. Sin manzanas cerca → vacío.

Uso:
    python -m src.censo_dane                 # shapefiles → data/dane/manzanas_cnpv2018.parquet
    from src.censo_dane import features_censo
    extra = features_censo(df)               # SOLO las columnas nuevas (latitud/longitud)
"""
from __future__ import annotations
import glob

import numpy as np
import pandas as pd

from src.config import BASE_DIR

DIR = BASE_DIR / "data" / "dane"
PARQUET = DIR / "manzanas_cnpv2018.parquet"
RADIO_M = 250
ESTRATOS = [f"TP19_EE_E{i}" for i in range(1, 7)]                     # viviendas por estrato 1–6
EDUCACION = ["TP51PRIMAR", "TP51SECUND", "TP51SUPERI", "TP51POSTGR", "TP51_13_ED", "TP51_99_ED"]
CONTEOS = (["TVIVIENDA", "TP14_1_TIP", "TP14_2_TIP", "TP15_1_OCU", "TP15_4_OCU", "TP19_INTE1", "TP19_INTE2",
            "TP19_GAS_1", "TP19_GAS_2", "TP27_PERSO", "AREA"] + ESTRATOS + EDUCACION)


def preparar() -> pd.DataFrame:
    """Shapefiles del DANE → parquet con centroide, conteos e IPM por manzana."""
    import duckdb
    con = duckdb.connect()
    con.execute("LOAD spatial;")
    mgn = glob.glob(str(DIR / "mgn_manzana" / "MGN_ANM_MANZANA.shp"))[0]
    ipm = glob.glob(str(DIR / "ipm_manzana" / "VULNRB_IPMxMZ.shp"))[0]
    cols = ", ".join(f'"{c}"' for c in CONTEOS)
    m = con.execute(f'SELECT COD_DANE_A AS cod, MPIO_CDPMP AS mpio, LATITUD AS lat, LONGITUD AS lon, {cols} '
                    f"FROM ST_Read('{mgn}')").df()
    i = con.execute(f"SELECT COD_DANE AS cod, ipm FROM ST_Read('{ipm}')").df()
    m = m.merge(i, on="cod", how="left")
    m.to_parquet(PARQUET, index=False)
    print(f"✅ {len(m):,} manzanas → {PARQUET.relative_to(BASE_DIR)} ({PARQUET.stat().st_size / 1e6:.0f} MB) · "
          f"con IPM: {m['ipm'].notna().mean():.0%}")
    return m


def _km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.column_stack([lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57])


def features_censo(df: pd.DataFrame, manzanas: pd.DataFrame | None = None, radio_m: float = RADIO_M) -> pd.DataFrame:
    """Proporciones del censo 2018 en las manzanas a menos de `radio_m` de cada aviso."""
    from scipy.spatial import cKDTree
    m = pd.read_parquet(PARQUET) if manzanas is None else manzanas
    m = m.dropna(subset=["lat", "lon"])
    lat, lon = ("latitud", "longitud") if "latitud" in df else ("Latitud", "Longitud")
    ok = (df[lat].notna() & df[lon].notna()).to_numpy()
    pts = _km(df.loc[ok, lat].astype(float).to_numpy(), df.loc[ok, lon].astype(float).to_numpy())
    arbol = cKDTree(_km(m["lat"].to_numpy(), m["lon"].to_numpy()))
    vecinas = arbol.query_ball_point(pts, r=radio_m / 1000, workers=-1)

    # Sumar los conteos de las manzanas vecinas (matriz dispersa aviso × manzana)
    from scipy.sparse import csr_matrix
    filas = np.repeat(np.arange(len(vecinas)), [len(v) for v in vecinas])
    cols = np.concatenate([np.asarray(v, dtype=int) for v in vecinas]) if len(filas) else np.array([], dtype=int)
    W = csr_matrix((np.ones(len(filas)), (filas, cols)), shape=(len(vecinas), len(m)))
    S = pd.DataFrame(W @ m[CONTEOS].fillna(0).to_numpy(), columns=CONTEOS)
    ipm = m["ipm"].to_numpy()
    con_ipm = np.nan_to_num(ipm)
    viv_ipm = W @ (m["TVIVIENDA"].fillna(0).to_numpy() * ~np.isnan(ipm))
    S["ipm"] = np.where(viv_ipm > 0, (W @ (con_ipm * m["TVIVIENDA"].fillna(0).to_numpy())) / np.maximum(viv_ipm, 1), np.nan)

    div = lambda a, b: np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)
    estratos = S[ESTRATOS].to_numpy()
    edu = S[EDUCACION].sum(axis=1).to_numpy()
    res = pd.DataFrame({
        "censo_estrato_medio": div(estratos @ np.arange(1, 7), estratos.sum(axis=1)),
        "censo_pct_superior": div(S["TP51SUPERI"] + S["TP51POSTGR"], edu),
        "censo_pct_posgrado": div(S["TP51POSTGR"], edu),
        "censo_pct_apartamentos": div(S["TP14_2_TIP"], S["TVIVIENDA"]),
        "censo_pct_desocupadas": div(S["TP15_4_OCU"], S["TVIVIENDA"]),
        "censo_pct_internet": div(S["TP19_INTE1"], S["TP19_INTE1"] + S["TP19_INTE2"]),
        "censo_pct_gas": div(S["TP19_GAS_1"], S["TP19_GAS_1"] + S["TP19_GAS_2"]),
        "censo_personas_ha": div(S["TP27_PERSO"], S["AREA"] / 10_000),
        "censo_viviendas": S["TVIVIENDA"].to_numpy(),
        "censo_ipm": S["ipm"].to_numpy(),
    })
    res.loc[res["censo_viviendas"] == 0] = np.nan          # sin viviendas censadas cerca: no hay dato
    out = pd.DataFrame(index=df.index, columns=res.columns, dtype=float)
    out.loc[ok] = res.to_numpy()
    return out


if __name__ == "__main__":
    preparar()
