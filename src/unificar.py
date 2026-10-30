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

Estado unificado (data/estado_unificado/ y estado_unificado/ en R2), al final del barrido de Metrocuadrado:
  avisos.parquet     1 fila por aviso que haya estado en el registro: portal, id, url, precio, su inmueble
                     (id_unificado), activo y desde cuándo está en ese inmueble
  inmuebles.parquet  1 fila por inmueble que haya existido: primera_vez (la del aviso más antiguo),
                     ultima_vez, activo, avisos, portales, ficha y precios; fusionado_en si se juntó con otro
  historial/fecha=…  los avisos que ese día entraron a un inmueble, cambiaron de inmueble o salieron
  meta.json          último día, último id y resumen del día (lo lee el monitor)

`id_unificado` es un número que no cambia aunque entren o salgan avisos de su grupo. Cada día, con los
grupos de hoy y el id que tenía cada aviso:
  - cada id pasa al grupo de hoy que tiene más de sus avisos (si el grupo se parte, el resto recibe otro);
  - si a un grupo le llegan varios ids, se queda con el más antiguo (el menor) y los demás quedan
    `fusionado_en` él;
  - un grupo sin ningún aviso conocido recibe un id nuevo (consecutivos, del aviso más antiguo al más nuevo);
  - un inmueble sin avisos activos queda inactivo con su id, y lo recupera si vuelve uno de sus avisos.

Uso:
    python -m src.unificar [--ciudad Cali]              # foto del día en data/unificado/ (análisis, sin estado)
    python -m src.unificar --fecha 2026-10-29 [--bajar] [--subir]   # el CI: estado unificado del día
"""
from __future__ import annotations
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.deduplicar import _km, _texto, diferencia_area, matriz_texto, puntaje, señales

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
ESTADO = BASE_DIR / "data" / "estado_unificado"
R2 = "estado_unificado"


def cargar(ciudad: str | None = None) -> pd.DataFrame:
    """Avisos de vivienda activos de los dos portales, en una sola tabla con `portal` y `clave`."""
    from src.consolidar import rutas
    partes = []
    for portal, raiz in PORTALES.items():
        a = pd.read_parquet(rutas(raiz).avisos, columns=COLUMNAS + ["estado_aviso"])
        a = a[(a["estado_aviso"] == "activo") & a["tipo"].isin(TIPOS)]
        if ciudad:
            a = a[a["ciudad"] == ciudad]
        partes.append(a[COLUMNAS].assign(portal=portal, tipo=a["tipo"].map(TIPOS)))
    d = pd.concat(partes, ignore_index=True)
    d["id_inmueble"] = d["id_inmueble"].astype(str)
    d["anunciante_id"] = d["anunciante_id"].astype(str).where(d["anunciante_id"].notna())
    d["primera_vez"] = d["primera_vez"].astype(str)
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


def grupos(d: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """(grupo de cada aviso, pares puntuados)."""
    t0 = time.time()
    i, j = candidatos(d)
    print(f"{len(d):,} avisos · {len(i):,} pares candidatos · {time.time() - t0:.0f} s", flush=True)
    T = matriz_texto(_texto(d))
    P = señales(d, d, i, j, T, T)
    P["puntaje"] = puntaje(P).astype(np.float32)
    g = agrupar(d, P)
    print(f"{len(set(g)):,} inmuebles · {time.time() - t0:.0f} s", flush=True)
    return g, P


def resumir(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por inmueble (`id_unificado`) con sus avisos, portales, ficha y precios."""
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    return d.assign(precio=num("precio"), m2=num("m2"), habitaciones=num("habitaciones"),
                    latitud=num("latitud"), longitud=num("longitud")).groupby("id_unificado").agg(
        n_avisos=("clave", "size"), portales=("portal", lambda s: "+".join(sorted(set(s)))),
        primera_vez=("primera_vez", "min"), tipo=("tipo", "first"), ciudad=("ciudad", "first"),
        barrio=("barrio", "first"), latitud=("latitud", "mean"), longitud=("longitud", "mean"), m2=("m2", "median"),
        habitaciones=("habitaciones", "median"), precio_min=("precio", "min"),
        precio_mediano=("precio", "median"), precio_max=("precio", "max")).reset_index()


def unificar(ciudad: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Foto del día, sin estado: (avisos con su id_unificado, inmuebles, pares puntuados). Aquí el id es
    la clave del aviso más antiguo del grupo ("portal:id")."""
    d = cargar(ciudad)
    g, P = grupos(d)
    orden = d.sort_values(["primera_vez", "clave"])
    rep = orden.groupby(g[orden.index])["clave"].first()                  # el aviso más antiguo del grupo
    d["id_unificado"] = rep.reindex(g).to_numpy()
    d["avisos_en_grupo"] = d.groupby("id_unificado")["clave"].transform("size")
    return d.drop(columns=["descripcion"]), resumir(d), P


# ── Estado unificado: id estable de un día a otro ─────────────────────────────


def cargar_estado() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if (ESTADO / "avisos.parquet").exists():
        return (pd.read_parquet(ESTADO / "avisos.parquet"), pd.read_parquet(ESTADO / "inmuebles.parquet"),
                json.loads((ESTADO / "meta.json").read_text()))
    return (pd.DataFrame(columns=["clave", "id_unificado", "activo", "desde"]),
            pd.DataFrame(columns=["id_unificado", "primera_vez", "activo", "fusionado_en"]),
            {"ultimo_dia": None, "dias": [], "ultimo_id": 0})


def asignar(d: pd.DataFrame, g: np.ndarray, avisos: pd.DataFrame, inmuebles: pd.DataFrame,
            ultimo_id: int) -> tuple[np.ndarray, pd.DataFrame, int, int]:
    """id estable de cada aviso de hoy con las reglas del docstring del módulo.
    (ids, fusiones viejo → en, ids partidos, nuevo último id)."""
    fusion = dict(inmuebles.dropna(subset=["fusionado_en"])[["id_unificado", "fusionado_en"]].astype(int).to_numpy())

    def vigente(i: int) -> int:                 # el id en que terminó una cadena de fusiones
        while i in fusion:
            i = fusion[i]
        return i
    viejo = d["clave"].map(avisos.set_index("clave")["id_unificado"]).map(vigente, na_action="ignore")
    t = pd.DataFrame({"grupo": g, "viejo": viejo.to_numpy()}).dropna()
    n = t.value_counts().rename("n").reset_index().sort_values(["viejo", "n", "grupo"], ascending=[True, False, True])
    dueño = n.drop_duplicates("viejo").set_index("viejo")["grupo"]        # cada id viejo → su grupo de hoy
    gana = dueño.reset_index().groupby("grupo")["viejo"].min()             # cada grupo → el id más antiguo
    fusiones = dueño.map(gana).rename("en").reset_index().query("viejo != en").astype(int)
    orden = pd.Series(g, index=d.index).loc[d.sort_values(["primera_vez", "clave"]).index]
    sin_id = [x for x in pd.unique(orden) if x not in gana.index]
    gana = pd.concat([gana, pd.Series(np.arange(ultimo_id + 1, ultimo_id + 1 + len(sin_id)), index=sin_id)])
    partidos = int((n.groupby("viejo").size() > 1).sum())
    return gana.reindex(g).astype("int64").to_numpy(), fusiones, partidos, ultimo_id + len(sin_id)


def actualizar(fecha: str, d: pd.DataFrame | None = None) -> dict:
    """Aplica los avisos activos de `fecha` sobre el estado unificado y escribe el historial del día.
    `d`: la tabla de cargar() (por defecto, los estados de los dos portales en data/)."""
    from src.almacen import COMPRESION
    from src.fincaraiz_http import guardar_parquet
    avisos, inmuebles, meta = cargar_estado()
    if meta["ultimo_dia"] and fecha <= meta["ultimo_dia"]:
        raise ValueError(f"El {fecha} ya está procesado (último: {meta['ultimo_dia']}).")
    d = cargar() if d is None else d.reset_index(drop=True)
    g, _ = grupos(d)
    ids, fusiones, partidos, meta["ultimo_id"] = asignar(d, g, avisos, inmuebles, meta["ultimo_id"])
    d["id_unificado"] = ids

    # Avisos: los de hoy activos con su id; los demás, inactivos con el último que tuvieron
    antes = avisos.set_index("clave")
    id_antes = d["clave"].map(antes["id_unificado"])
    activo_antes = d["clave"].map(antes["activo"]).fillna(False).astype(bool)
    hoy = d[["clave", "portal", "id_inmueble", "url", "precio", "id_unificado"]].assign(
        activo=True, desde=np.where(id_antes == d["id_unificado"], d["clave"].map(antes["desde"]), fecha))
    salen = avisos[avisos["activo"].astype(bool) & ~avisos["clave"].isin(d["clave"])]
    avisos = pd.concat([hoy, avisos[~avisos["clave"].isin(d["clave"])].assign(activo=False)], ignore_index=True)
    cambia = activo_antes & (id_antes != d["id_unificado"])
    hist = pd.concat([
        pd.DataFrame({"clave": d["clave"], "evento": "entra", "id_antes": id_antes, "id_despues": d["id_unificado"]})[~activo_antes],
        pd.DataFrame({"clave": d["clave"], "evento": "cambia", "id_antes": id_antes, "id_despues": d["id_unificado"]})[cambia],
        pd.DataFrame({"clave": salen["clave"], "evento": "sale", "id_antes": salen["id_unificado"], "id_despues": np.nan}),
    ], ignore_index=True).assign(fecha=fecha)

    # Inmuebles: los de hoy activos; los demás inactivos (y fusionado_en si se juntaron con otro)
    previo = inmuebles.set_index("id_unificado")
    nuevos = resumir(d).set_index("id_unificado")
    hoy_pv = nuevos["primera_vez"].astype(str)
    antes_pv = previo["primera_vez"].reindex(nuevos.index).fillna(hoy_pv).astype(str)
    nuevos["primera_vez"] = np.where(antes_pv < hoy_pv, antes_pv, hoy_pv)     # la más antigua
    nuevos = nuevos.assign(activo=True, ultima_vez=fecha, fusionado_en=np.nan)
    resto = previo.drop(nuevos.index, errors="ignore")
    era_activo = resto["activo"].astype(bool) & ~resto.index.isin(fusiones["viejo"])
    resto = resto.assign(activo=False)
    resto.loc[fusiones["viejo"].to_numpy(), "fusionado_en"] = fusiones["en"].to_numpy()
    cerrados = int(era_activo.sum())
    inmuebles = pd.concat([nuevos, resto]).reset_index().astype({"n_avisos": "int64"})

    ESTADO.mkdir(parents=True, exist_ok=True)
    (ESTADO / "historial").mkdir(exist_ok=True)
    guardar_parquet(avisos, ESTADO / "avisos.parquet", compression=COMPRESION)
    guardar_parquet(inmuebles, ESTADO / "inmuebles.parquet", compression=COMPRESION)
    guardar_parquet(hist, ESTADO / "historial" / f"fecha={fecha}.parquet", compression=COMPRESION)
    siguen = int(activo_antes.sum())
    multi = nuevos[nuevos["n_avisos"] > 1]
    resumen = {"fecha": fecha, "avisos_activos": len(d), "inmuebles_activos": len(nuevos),
               "con_2_o_mas_avisos": len(multi), "en_los_dos_portales": int(multi["portales"].str.contains(r"\+").sum()),
               "inmuebles_nuevos": int(len(nuevos) - nuevos.index.isin(previo.index).sum()),
               "inmuebles_cerrados": cerrados, "fusionados": len(fusiones), "partidos": partidos,
               "avisos_entran": int((~activo_antes).sum()), "avisos_salen": len(salen),
               "avisos_cambian": int(cambia.sum()),
               "estabilidad": round(1 - cambia.sum() / siguen, 4) if siguen else None}
    meta = meta | {"ultimo_dia": fecha, "dias": meta["dias"] + [fecha], "ultimo_resumen": resumen}
    (ESTADO / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, ensure_ascii=False))
    return resumen


def subir(fecha: str) -> None:
    """Estado (+ respaldo diario), historial del día y meta a R2."""
    from src import almacen
    for nombre in ("avisos.parquet", "inmuebles.parquet"):
        almacen.subir_archivo(ESTADO / nombre, f"{R2}/{nombre}")
        almacen.subir_archivo(ESTADO / nombre, f"respaldos/diario/fecha={fecha}/unificado_{nombre}")
    almacen.subir_archivo(ESTADO / "historial" / f"fecha={fecha}.parquet", f"{R2}/historial/fecha={fecha}.parquet")
    almacen.subir_archivo(ESTADO / "meta.json", f"{R2}/meta.json")


def bajar() -> None:
    """Trae de R2 el estado de los dos portales y el unificado (si existe) para continuar desde ahí."""
    from src import almacen, consolidar
    for raiz in PORTALES.values():
        consolidar.bajar(raiz)
    tmp = BASE_DIR / "data" / "_r2tmp"
    for nombre in ("avisos.parquet", "inmuebles.parquet", "meta.json"):
        almacen.traer(f"{R2}/{nombre}", tmp)
    if (tmp / R2 / "meta.json").exists():
        ESTADO.mkdir(parents=True, exist_ok=True)
        for nombre in ("avisos.parquet", "inmuebles.parquet", "meta.json"):
            (tmp / R2 / nombre).replace(ESTADO / nombre)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Un solo registro por inmueble (E8)")
    p.add_argument("--ciudad", help="foto del día: solo esta ciudad (por defecto, todo el país)")
    p.add_argument("--fecha", help="aplica el día al estado unificado (AAAA-MM-DD)")
    p.add_argument("--bajar", action="store_true", help="trae de R2 los estados antes de aplicar")
    p.add_argument("--subir", action="store_true", help="sube el estado unificado del día a R2")
    a = p.parse_args()
    if a.bajar:
        bajar()
    if a.fecha:
        try:
            actualizar(a.fecha)
        except ValueError as e:                      # p. ej. Metrocuadrado relanzado el mismo día
            print(f"⚠️  {e} No se aplica de nuevo.")
        else:
            if a.subir:
                subir(a.fecha)
    elif not a.bajar:
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
