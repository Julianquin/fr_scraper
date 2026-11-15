"""
Los movimientos del mercado por inmueble, con los dos portales (E8.2): un inmueble publicado en los dos portales, o
por varias inmobiliarias, cuenta una vez. Es un solo mercado: nada de lo que sale de aquí dice de qué portal viene.

    publicada   el inmueble apareció en el periodo: su primer aviso (en cualquier portal) es de entonces. FincaRaíz da
                la fecha de publicación; Metrocuadrado no, así que vale el día en que su barrido lo vio aparecer
                (`alta`); los que su barrido encontró ya publicados (`descubierto`) son viejos.
    retirada    salió del mercado en el periodo: ya no tiene ningún aviso activo, y el último se dio de baja entonces.
                Un aviso que sale de un portal mientras el inmueble sigue en el otro no es una retirada.
    bajada      bajó de precio en el periodo en cualquiera de sus avisos (≥ BAJA_MIN, como las bajadas del canal).
    cambio      «el precio de las mismas viviendas»: de los inmuebles en venta todo el periodo, el cambio de su precio
                pedido (el más marcado de sus avisos; casi siempre piden lo mismo en los dos portales).

El inmueble de cada aviso sale del estado unificado (src/unificar.py, `estado_unificado/` en R2: también los avisos
ya inactivos, con las fusiones resueltas). Sin él, cada aviso cuenta como un inmueble.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

PORTALES = (("venta", "fincaraiz"), ("m2_venta", "metrocuadrado"))


def _ids_unificados() -> pd.Series:
    """clave del aviso («fincaraiz:123») → inmueble, siguiendo las fusiones hasta el que sigue vivo."""
    from src.unificar import ESTADO
    if not (ESTADO / "avisos.parquet").exists():
        print("⚠️  Sin el estado unificado: cada aviso cuenta como un inmueble")
        return pd.Series(dtype="int64")
    av = pd.read_parquet(ESTADO / "avisos.parquet", columns=["clave", "id_unificado"])
    destino = pd.Series(dtype="float64")
    if (ESTADO / "inmuebles.parquet").exists():
        i = pd.read_parquet(ESTADO / "inmuebles.parquet", columns=["id_unificado", "fusionado_en"])
        destino = i.dropna(subset=["fusionado_en"]).set_index("id_unificado")["fusionado_en"].astype("int64")
    ids = av.set_index("clave")["id_unificado"].astype("int64")
    for _ in range(20):                         # cadenas de fusiones (a → b → c)
        siguiente = ids.map(destino)
        if siguiente.isna().all():
            break
        ids = siguiente.fillna(ids).astype("int64")
    return ids


def _historial(raiz: str, eventos: tuple[str, ...]) -> pd.DataFrame:
    from src.consolidar import rutas
    partes = [pd.read_parquet(p, columns=["fecha", "id_inmueble", "evento"])
              for p in sorted(rutas(raiz).historial.glob("fecha=*.parquet"))]
    if not partes:
        return pd.DataFrame(columns=["fecha", "id_inmueble", "evento"])
    h = pd.concat(partes, ignore_index=True)
    return h[h["evento"].isin(eventos)].assign(id_inmueble=lambda x: x["id_inmueble"].astype(str))


@lru_cache(maxsize=4)
def _avisos(ciudad: str) -> pd.DataFrame:
    """Los avisos de vivienda de la ciudad en los dos portales (también los retirados), con su inmueble, el día en que
    empezó (NaT y `viejo` si no se sabe: estaba antes de que lo viéramos), el de su última baja y su ubicación."""
    from src import oportunidades as op, zonas
    from src.consolidar import rutas
    ids = _ids_unificados()
    altas = op.altas_m2()
    partes = []
    for raiz, portal in PORTALES:
        r = rutas(raiz)
        if not r.avisos.exists():
            continue
        a = pd.read_parquet(r.avisos, columns=["id_inmueble", "ciudad", "tipo", "latitud", "longitud", "barrio",
                                               "estado_aviso", "fecha_publicacion"])
        aqui = (zonas.en_area(ciudad, a["latitud"], a["longitud"], a["ciudad"]) if ciudad in zonas.AREAS
                else a["ciudad"].map(zonas.canonica) == ciudad)
        a = a[aqui & a["tipo"].isin(op.TIPOS)].copy()
        a["id_inmueble"] = a["id_inmueble"].astype(str)
        a["portal"] = portal
        a["inicio"] = (pd.to_datetime(a["fecha_publicacion"], errors="coerce") if portal == "fincaraiz"
                       else a["id_inmueble"].map(altas))
        bajas = _historial(raiz, ("baja",)).groupby("id_inmueble")["fecha"].max()
        a["fin"] = pd.to_datetime(a["id_inmueble"].map(bajas))
        partes.append(a.drop(columns=["fecha_publicacion"]))
    a = pd.concat(partes, ignore_index=True)
    clave = a["portal"] + ":" + a["id_inmueble"]
    a["inmueble"] = clave.map(ids).astype("Int64").astype(str).where(clave.isin(ids.index), "aviso:" + clave)
    a["viejo"] = a["inicio"].isna()
    a["activo"] = a["estado_aviso"].eq("activo")
    return a


@lru_cache(maxsize=4)
def inmuebles(ciudad: str) -> pd.DataFrame:
    """Un registro por inmueble (activo o ya retirado): `inicio`, `viejo`, `activo`, `fin` (su última baja, si ya no
    tiene avisos activos), comuna y barrio oficial (de la ubicación de un aviso de FincaRaíz si lo tiene)."""
    from src import zonas
    a = _avisos(ciudad)
    g = a.groupby("inmueble")
    t = pd.DataFrame({"inicio": g["inicio"].min(), "viejo": g["viejo"].any(), "activo": g["activo"].any(),
                      "fin": g["fin"].max()})
    t.loc[t["activo"], "fin"] = pd.NaT
    lugar = a.sort_values("portal").drop_duplicates("inmueble").set_index("inmueble")    # fincaraiz antes
    t = t.join(lugar[["latitud", "longitud", "barrio"]])
    if zonas.tiene_zonas(ciudad):
        z = zonas.asignar_zonas(t.rename(columns={"latitud": "Latitud", "longitud": "Longitud", "barrio": "Barrio"}),
                                ciudad)
        t["comuna"], t["barrio_oficial"] = z["comuna"].to_numpy(), z["barrio_oficial"].to_numpy()
    else:
        t["comuna"] = t["barrio_oficial"] = None
    return t


def _precios(ciudad: str, desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.DataFrame:
    """Por inmueble, en el periodo: `factor` (precio al final ÷ al principio, el más marcado de sus avisos) y `baja`
    (la mayor bajada de un aviso, de BAJA_MIN a 50 %, como las bajadas del canal; también al republicar)."""
    from src import oportunidades as op
    a = _avisos(ciudad)
    ev = op._eventos()
    ev = ev[ev["fecha"].between(f"{desde:%Y-%m-%d}", f"{hasta:%Y-%m-%d}")]
    inm = a.drop_duplicates("id_inmueble").set_index("id_inmueble")["inmueble"]
    cp = ev[ev["evento"] == "cambio_precio"]
    f = (cp["precio_despues"] / cp["precio_antes"]).groupby(cp["id_inmueble"]).prod()
    f = pd.DataFrame({"factor": f, "inmueble": f.index.map(inm)}).dropna(subset=["inmueble"])
    factor = f.loc[(f["factor"] - 1).abs().groupby(f["inmueble"]).idxmax(), ["inmueble", "factor"]].set_index("inmueble")
    b = ev[ev["evento"].isin(["cambio_precio", "republicado"]) & (ev["precio_despues"] < ev["precio_antes"])]
    b = b.assign(baja=1 - b["precio_despues"] / b["precio_antes"], inmueble=b["id_inmueble"].map(inm))
    b = b[b["baja"].between(op.BAJA_MIN, 0.5)].dropna(subset=["inmueble"])
    return factor.join(b.groupby("inmueble")["baja"].max(), how="outer")


def periodo(ciudad: str, desde: pd.Timestamp, hasta: pd.Timestamp, por: str | None = None) -> pd.DataFrame:
    """Los movimientos de un periodo, de la ciudad (una fila, índice «total») o por `por` ("comuna" o
    "barrio_oficial"): publicadas, retiradas, bajadas y su bajada mediana; en_venta (activos hoy), en_venta_todo
    (activos hoy y ya publicados antes del periodo), bajaron y subieron (de esos, ≥ BAJA_MIN), cambio_mismas (cambio
    medio de su precio)."""
    from src import oportunidades as op
    desde, hasta = pd.Timestamp(desde), pd.Timestamp(hasta)
    t = inmuebles(ciudad).join(_precios(ciudad, desde, hasta))
    t["factor"] = t["factor"].fillna(1.0)
    t["publicada"] = ~t["viejo"] & t["inicio"].between(desde, hasta)
    t["retirada"] = ~t["activo"] & t["fin"].between(desde, hasta)
    t["bajada"] = t["baja"].notna()
    t["todo"] = t["activo"] & (t["viejo"] | (t["inicio"] < desde))
    clave = pd.Series("total", index=t.index) if por is None else t[por]

    def resumen(x: pd.DataFrame) -> pd.Series:
        m = x[x["todo"]]
        cambio = m["factor"] - 1
        return pd.Series({
            "publicadas": int(x["publicada"].sum()), "retiradas": int(x["retirada"].sum()),
            "bajadas": int(x["bajada"].sum()),
            "bajada_mediana": float(x.loc[x["bajada"], "baja"].median()) if x["bajada"].any() else np.nan,
            "en_venta": int(x["activo"].sum()), "en_venta_todo": len(m),
            "bajaron": int((cambio <= -op.BAJA_MIN).sum()), "subieron": int((cambio >= op.BAJA_MIN).sum()),
            "cambio_mismas": float(cambio.mean()) if len(m) else np.nan,
            "bajada_mediana_mismas": float(-cambio[cambio <= -op.BAJA_MIN].median())
            if (cambio <= -op.BAJA_MIN).any() else np.nan,
        })

    return t.dropna(subset=[] if por is None else [por]).groupby(clave.dropna()).apply(resumen)
