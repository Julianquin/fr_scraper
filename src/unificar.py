"""
Un solo registro por inmueble (bloque E8): agrupa los avisos de vivienda activos que son el mismo
inmueble, estén en dos portales (FincaRaíz y Metrocuadrado) o varias veces en el mismo portal (varias
inmobiliarias), y les da un `id_unificado`.

Cómo:
  1. Candidatos: mismo tipo, a menos de RADIO_KM, área parecida (AREA_MAX, la mejor combinación de área
     total y construida) y precio parecido (PRECIO_MAX). En Cali esos filtros dejan el 27% de los
     candidatos y el 99,3% de los pares buenos (29-oct). No se comparan dos avisos del mismo anunciante
     en el mismo portal: casi siempre son unidades distintas de un edificio o un proyecto.
  2. Señales y puntaje de src/deduplicar.py: texto, ficha, distancia y precio; piso distinto como veto,
     habitaciones y baños distintos a la vez como resta, y con texto débil hace falta precio idéntico.
  3. Grupos: se unen los pares de mayor a menor puntaje (desde UMBRAL), y dos grupos solo se juntan si
     TODOS sus avisos se parecen entre sí al menos UMBRAL_TODOS (no basta un par: un aviso parecido a dos
     edificios encadenaría inmuebles distintos). Además, un grupo no puede tener dos avisos del mismo
     anunciante en el mismo portal, dos pisos distintos ni más de GRUPO_MAX avisos.

Salida (data/unificado/, no se versiona):
  avisos.parquet     una fila por aviso: portal, id_inmueble, id_unificado y avisos de su grupo
  inmuebles.parquet  una fila por inmueble: n_avisos, portales, precio mínimo, mediano y máximo, ficha

`id_unificado` = el aviso más antiguo del grupo ("portal:id"). Todavía no es estable de un día a otro:
eso llega con el estado unificado (BACKLOG E8).

Uso:
    python -m src.unificar [--ciudad Cali]
"""
from __future__ import annotations
import argparse
import time

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.deduplicar import _km, _texto, diferencia_area, matriz_texto, puntaje, señales
from src.zonas import canonica

PORTALES = {"fincaraiz": "venta", "metrocuadrado": "m2_venta"}          # portal → raíz de consolidar
TIPOS = {"Apartamento": "Apartamento", "Apartaestudio": "Apartamento", "Casa": "Casa"}   # M2 no separa apartaestudios
COLUMNAS = ["id_inmueble", "url", "tipo", "precio", "m2", "m2_construido", "habitaciones", "banos", "estrato",
            "piso", "latitud", "longitud", "ciudad", "barrio", "anunciante_id", "primera_vez", "descripcion"]
RADIO_KM = 0.5
AREA_MAX = 0.15
PRECIO_MAX = 0.30
UMBRAL = 0.65
UMBRAL_TODOS = 0.5      # el par más débil del grupo resultante (vínculo completo)
GRUPO_MAX = 6
SALIDA = BASE_DIR / "data" / "unificado"


def cargar(ciudad: str | None = None) -> pd.DataFrame:
    """Avisos de vivienda activos de los dos portales, en una sola tabla con `portal` y `clave`."""
    from src.consolidar import rutas
    partes = []
    for portal, raiz in PORTALES.items():
        a = pd.read_parquet(rutas(raiz).avisos, columns=COLUMNAS + ["estado_aviso"])
        a = a[(a["estado_aviso"] == "activo") & a["tipo"].isin(TIPOS)]
        if ciudad:                      # Metrocuadrado escribe "Bogotá D.C." y "Cartagena de Indias"
            a = a[a["ciudad"].map(canonica) == canonica(ciudad)]
        partes.append(a[COLUMNAS].assign(portal=portal, tipo=a["tipo"].map(TIPOS)))
    d = pd.concat(partes, ignore_index=True)
    d["id_inmueble"] = d["id_inmueble"].astype(str)
    d["anunciante_id"] = d["anunciante_id"].astype(str).where(d["anunciante_id"].notna())
    d["clave"] = d["portal"] + ":" + d["id_inmueble"]
    return d


def candidatos(d: pd.DataFrame, lote: int = 5_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Pares (i < j) que vale la pena comparar: mismo tipo, cerca, área y precio parecidos, y no del mismo
    anunciante en el mismo portal. Los filtros van por tandas para no llenar la memoria."""
    from scipy.spatial import cKDTree
    X = _km(d)
    ok = np.isfinite(X).all(axis=1)
    precio = pd.to_numeric(d["precio"], errors="coerce").where(lambda v: v > 0).to_numpy()
    portal, anunciante = d["portal"].to_numpy(), d["anunciante_id"].to_numpy()
    ii, jj = [], []
    for tipo in d["tipo"].unique():
        idx = np.flatnonzero((d["tipo"] == tipo).to_numpy() & ok)
        p = cKDTree(X[idx]).query_pairs(RADIO_KM, output_type="ndarray")
        for k in range(0, len(p), lote):
            i, j = idx[p[k:k + lote, 0]], idx[p[k:k + lote, 1]]
            with np.errstate(all="ignore"):
                area = diferencia_area(d, d, i, j)
                dp = np.abs(np.log(precio[i] / precio[j]))
            mismo = (portal[i] == portal[j]) & (anunciante[i] == anunciante[j]) & pd.notna(anunciante[i])
            sirve = ~(area > AREA_MAX) & ~(dp > PRECIO_MAX) & ~mismo       # sin dato (NaN) no descarta
            ii.append(i[sirve])
            jj.append(j[sirve])
    return np.concatenate(ii), np.concatenate(jj)


def agrupar(d: pd.DataFrame, P: pd.DataFrame, umbral: float = UMBRAL) -> np.ndarray:
    """Grupo de cada aviso (índice de su representante), uniendo los pares aceptados de mayor a menor
    puntaje con las restricciones del docstring del módulo."""
    padre = np.arange(len(d))
    miembros = {k: [k] for k in range(len(d))}
    anun = {k: {(p, a)} for k, (p, a) in enumerate(zip(d["portal"], d["anunciante_id"])) if pd.notna(a)}
    pisos = {k: {v} for k, v in enumerate(pd.to_numeric(d["piso"], errors="coerce").where(lambda s: s > 0))
             if pd.notna(v)}
    parecidos = P[P["puntaje"] >= UMBRAL_TODOS]
    parecidos = set(zip(parecidos[["i", "j"]].min(axis=1), parecidos[["i", "j"]].max(axis=1)))

    def raiz(k: int) -> int:
        while padre[k] != k:
            padre[k] = padre[padre[k]]
            k = padre[k]
        return k

    acept = P[P["puntaje"] >= umbral].sort_values("puntaje", ascending=False)
    for i, j in zip(acept["i"].to_numpy(), acept["j"].to_numpy()):
        ri, rj = raiz(i), raiz(j)
        if ri == rj or len(miembros[ri]) + len(miembros[rj]) > GRUPO_MAX:
            continue
        ai, aj, pi, pj = anun.get(ri, set()), anun.get(rj, set()), pisos.get(ri, set()), pisos.get(rj, set())
        if ai & aj or len(pi | pj) > 1:
            continue
        if not all((min(a, b), max(a, b)) in parecidos for a in miembros[ri] for b in miembros[rj]):
            continue
        padre[rj] = ri
        miembros[ri] += miembros.pop(rj)
        anun[ri], pisos[ri] = ai | aj, pi | pj
    return np.array([raiz(k) for k in range(len(d))])


def unificar(ciudad: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(avisos con su id_unificado, inmuebles, pares puntuados)."""
    t0 = time.time()
    d = cargar(ciudad)
    i, j = candidatos(d)
    print(f"{len(d):,} avisos · {len(i):,} pares candidatos · {time.time() - t0:.0f} s", flush=True)
    T = matriz_texto(_texto(d))
    P = señales(d, d, i, j, T, T)
    P["puntaje"] = puntaje(P).astype(np.float32)
    print(f"señales y puntaje · {time.time() - t0:.0f} s", flush=True)
    g = agrupar(d, P)
    d["primera_vez"] = d["primera_vez"].astype(str)
    orden = d.sort_values(["primera_vez", "clave"])
    rep = orden.groupby(g[orden.index])["clave"].first()                  # el aviso más antiguo del grupo
    d["id_unificado"] = rep.reindex(g).to_numpy()
    d["avisos_en_grupo"] = d.groupby("id_unificado")["clave"].transform("size")
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    inm = d.assign(precio=num("precio"), m2=num("m2"), habitaciones=num("habitaciones"),
                   latitud=num("latitud"), longitud=num("longitud")).groupby("id_unificado").agg(
        n_avisos=("clave", "size"), portales=("portal", lambda s: "+".join(sorted(set(s)))),
        tipo=("tipo", "first"), ciudad=("ciudad", "first"), barrio=("barrio", "first"),
        latitud=("latitud", "mean"), longitud=("longitud", "mean"), m2=("m2", "median"),
        habitaciones=("habitaciones", "median"), precio_min=("precio", "min"),
        precio_mediano=("precio", "median"), precio_max=("precio", "max")).reset_index()
    print(f"{len(inm):,} inmuebles · {time.time() - t0:.0f} s", flush=True)
    return d.drop(columns=["descripcion"]), inm, P


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Un solo registro por inmueble (E8)")
    p.add_argument("--ciudad", help="solo esta ciudad (por defecto, todo el país)")
    a = p.parse_args()
    avisos, inmuebles, _ = unificar(a.ciudad)
    SALIDA.mkdir(parents=True, exist_ok=True)
    cols = ["clave", "portal", "id_inmueble", "id_unificado", "avisos_en_grupo", "tipo", "ciudad", "barrio",
            "precio", "m2", "habitaciones", "anunciante_id", "url"]
    avisos[cols].to_parquet(SALIDA / "avisos.parquet", index=False)
    inmuebles.to_parquet(SALIDA / "inmuebles.parquet", index=False)
    por_portal = avisos.groupby("portal").size().to_dict()
    multi = inmuebles[inmuebles["n_avisos"] > 1]
    print(f"✅ {len(avisos):,} avisos ({por_portal}) → {len(inmuebles):,} inmuebles "
          f"({1 - len(inmuebles) / len(avisos):.1%} menos) · {len(multi):,} con 2+ avisos, de ellos "
          f"{(multi['portales'] == 'fincaraiz+metrocuadrado').sum():,} en los dos portales")
