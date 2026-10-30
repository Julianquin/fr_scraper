"""
Estado + historia del mercado a partir de los barridos diarios (bloques E1/E2; arriendos en E3).

Cada barrido lee ~170.000 avisos, pero de un día a otro casi todo es igual. En vez de guardar una
copia diaria, se mantiene:

  avisos     estado/avisos.parquet                 1 fila por aviso que haya existido alguna vez:
                                                   datos más recientes + primera_vez, ultima_vez,
                                                   estado_aviso (activo | retirado), fecha_baja
  historial  estado/historial/fecha=AAAA-MM-DD.parquet   solo lo que cambió ese día (nunca se reescribe,
                                                   salvo la corrección única del 30-oct, ver abajo):
                                                   alta · descubierto · republicado · cambio_precio ·
                                                   cambio_administracion · cambio_descripcion ·
                                                   baja · reaparece · reactivado

Reglas:
  - Alta vs. descubierto: cada barrido ve ~93% del país y el ~7% que falta cambia de un día a otro.
    Un aviso que vemos por primera vez es "alta" solo si se publicó desde el barrido anterior; si
    ya estaba publicado, es "descubierto" (el 26-oct fueron 588 altas y 4.212 descubiertos).
  - Baja: el aviso falta en 3 barridos seguidos (con 2 habría muchas bajas falsas por ese ~7%
    variable). La fecha de baja es la del primer barrido en que faltó. Las "reaparece" después de
    una baja miden cuántas bajas fueron falsas.
  - Reactivado: un aviso que el rescate del sistema viejo (src/rescatar.py) dejó como retirado, sin
    fecha de baja, y que el barrido vuelve a encontrar. No dice nada de la regla de bajas: va aparte
    para no inflar la tasa de "reaparece" (el 27-oct fueron 395, ninguno con baja del barrido).
  - Republicado: un aviso nuevo que es uno que faltó hoy (o que se retiró hace menos de
    REPUBLICADO_DIAS) vuelto a publicar con otro id: mismo anunciante y `codigo_anunciante`, uno a uno,
    tipo equivalente y área parecida. No cuenta como alta: hereda la `primera_vez` del viejo (los días en
    el mercado siguen contando) y el evento guarda el precio antes y después y, en `texto_antes`, el id
    viejo. El viejo queda retirado sin evento de baja y con `republicado_como` = el id nuevo. El 30-oct
    una inmobiliaria de Medellín republicó así ~9.000 avisos (BACKLOG E2); en un día normal son 10–50.
  - Protección: si el barrido no es completo, no se cuentan ausencias ese día (un grupo caído no debe
    generar miles de bajas falsas). Completo = cubre al menos el 90% de lo que declara el portal
    (`reporte.json`) y trae al menos el 80% de nuestros activos; sin reporte, el 90% de los activos.
    Hasta el 30-oct se comparaba solo con los activos, y una republicación masiva la dejaba saltando
    cada día (los avisos viejos nunca llegaban a baja porque no se contaban sus ausencias).
  - Cada fecha se procesa una sola vez y en orden.

Arriendos (`--raiz arriendo`): mismas reglas, con su propio estado (data/estado_arriendo/ y
estado_arriendo/ en R2). En arriendos, `precio` es el canon mensual.

Metrocuadrado (`--raiz m2_venta`, src/metrocuadrado_http.py): mismas reglas y estado propio
(data/estado_m2_venta/ y estado_m2_venta/ en R2). El portal no publica la fecha de publicación, así
que todo lo nuevo desde el barrido anterior cuenta como alta (con ~99% de cobertura, el ~1% variable
puede colarse como alta).

Uso:
    python -m src.consolidar --fecha 2026-10-27 [--subir]   # un barrido (el CI, tras juntar)
    python -m src.consolidar --bajar                         # solo trae el estado de R2
    python -m src.consolidar --reconstruir                   # rehace todo con los barridos locales
    python -m src.consolidar --raiz arriendo …               # lo mismo para arriendos
    python -m src.consolidar --raiz arriendo --corregir-republicados 2026-10-30 --bajar --subir   # una vez
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.config import BASE_DIR
from src.fincaraiz_http import guardar_parquet
from src.scraper_http import RAICES, carpetas

RAICES_ESTADO = (*RAICES, "m2_venta")       # m2_venta: Metrocuadrado (src/metrocuadrado_http.py)


class Rutas(NamedTuple):
    raiz: str
    estado: Path
    avisos: Path
    historial: Path
    meta: Path
    r2: str             # prefijo en R2
    respaldo: str       # nombre del archivo en respaldos/{diario,semanal}/fecha=…/


def rutas(raiz: str = "venta") -> Rutas:
    """Venta conserva las rutas con que arrancó E1; arriendo usa el sufijo _arriendo."""
    sufijo = "" if raiz == "venta" else f"_{raiz}"
    estado = BASE_DIR / "data" / f"estado{sufijo}"
    return Rutas(raiz, estado, estado / "avisos.parquet", estado / "historial", estado / "meta.json",
                 f"estado{sufijo}", "avisos_http.parquet" if raiz == "venta" else f"{raiz}_http.parquet")


# Las de venta, como constantes (las usan src/rescatar.py y src/fuente_http.py)
ESTADO, AVISOS, HISTORIAL, META = rutas("venta")[1:5]
VIGILADOS = {"precio": "cambio_precio", "administracion": "cambio_administracion"}
AUSENCIAS_PARA_BAJA = 3
MIN_COBERTURA_PORTAL = 0.90     # avisos únicos / total que declara el portal (reporte.json)
MIN_COBERTURA_RELATIVA = 0.90   # vistos / activos propios, cuando no hay reporte
PISO_RELATIVO = 0.80            # vistos / activos propios, aunque el portal diga que está completo
REPUBLICADO_DIAS = 14           # un aviso retirado hace menos que esto todavía puede volver republicado
TIPOS_EQUIVALENTES = {"Apartaestudio": "Apartamento"}
# `estado` (sin sufijo) es el estado físico del inmueble ("Usado"…); el de control es `estado_aviso`
CONTROL = ["primera_vez", "ultima_vez", "estado_aviso", "fecha_baja", "ausencias", "primera_ausencia"]


def ruta_barrido(fecha: str, raiz: str = "venta") -> Path:
    """El crudo del barrido de `fecha`: el que deja `scraper_http juntar` o el bajado de R2."""
    _, crudo = carpetas(raiz)
    for p in (crudo / fecha / f"{raiz}.parquet",
              BASE_DIR / "data" / "r2" / "raw" / f"{raiz}_http" / f"fecha={fecha}" / "scrape.parquet"):
        if p.exists():
            return p
    raise FileNotFoundError(f"No hay barrido local de {raiz} del {fecha}")


def fechas_locales(raiz: str = "venta") -> list[str]:
    _, crudo = carpetas(raiz)
    fechas = {p.parent.name for p in crudo.glob(f"*/{raiz}.parquet")}
    fechas |= {p.parent.name.split("=")[1] for p in
               (BASE_DIR / "data" / "r2" / "raw" / f"{raiz}_http").glob("fecha=*/scrape.parquet")}
    return sorted(fechas)


def _texto(s: pd.Series) -> pd.Series:
    """Normaliza espacios para que un cambio de formato no cuente como cambio de descripción."""
    return s.fillna("").astype(str).str.split().str.join(" ")


def _evento(fecha, ids, evento, antes=None, despues=None, texto_antes=None, fuente="barrido") -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame({
        "fecha": fecha, "id_inmueble": list(ids), "evento": evento,
        "valor_antes": antes if antes is not None else [np.nan] * n,
        "valor_despues": despues if despues is not None else [np.nan] * n,
        "texto_antes": texto_antes if texto_antes is not None else [None] * n,
        "fuente": fuente,       # barrido | scraper_viejo | foto_2025 (src/rescatar.py)
    })


def _cobertura(fecha: str, raiz: str) -> float | None:
    """Cobertura del barrido frente a lo que declara el portal (su reporte.json), si está."""
    try:
        reporte = ruta_barrido(fecha, raiz).parent / "reporte.json"
    except FileNotFoundError:
        return None
    return json.loads(reporte.read_text()).get("cobertura") if reporte.exists() else None


def _limpio(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip()
    return t.where(s.notna() & ~t.isin(["", "0", "nan", "None"]))


def republicaciones(viejos: pd.DataFrame, nuevos: pd.DataFrame) -> pd.DataFrame:
    """Pares (viejo, nuevo) de ids del mismo aviso vuelto a publicar: mismo anunciante y
    codigo_anunciante, uno a uno (un código repetido dentro del anunciante no sirve: hay quien usa el
    nombre del proyecto), tipo equivalente y área ±15% si las dos la tienen. El precio puede cambiar."""
    vacio = pd.DataFrame({"viejo": pd.Series(dtype=str), "nuevo": pd.Series(dtype=str)})
    if "codigo_anunciante" not in nuevos or viejos.empty or nuevos.empty:
        return vacio
    clave = ["anunciante_id", "_codigo"]

    def preparar(d: pd.DataFrame) -> pd.DataFrame:
        d = d.assign(_codigo=_limpio(d["codigo_anunciante"])).dropna(subset=clave)
        return d[~d.duplicated(clave, keep=False)]
    m = preparar(viejos).rename_axis("viejo").reset_index().merge(
        preparar(nuevos).rename_axis("nuevo").reset_index(), on=clave, suffixes=("_v", "_n"))
    if m.empty:
        return vacio
    tipo = m["tipo_v"].replace(TIPOS_EQUIVALENTES) == m["tipo_n"].replace(TIPOS_EQUIVALENTES)
    av, an = (pd.to_numeric(m[c], errors="coerce").where(lambda x: x > 0) for c in ("m2_v", "m2_n"))
    with np.errstate(all="ignore"):
        area = ~(np.abs(np.log(av / an)) > 0.15)                 # sin área no descarta
    return m.loc[tipo & area, ["viejo", "nuevo"]].reset_index(drop=True)


def _primer_dia_sin_ver(ultima_vez: pd.Series, barridos: list[str]) -> np.ndarray:
    """El primer barrido posterior a `ultima_vez`: el día en que el aviso dejó de estar."""
    b = np.array(sorted(barridos))
    i = np.searchsorted(b, ultima_vez.astype(str).to_numpy(), side="right")
    return np.where(i < len(b), b[np.minimum(i, len(b) - 1)], None)


def cargar_estado(r: Rutas = rutas()) -> tuple[pd.DataFrame, dict]:
    if r.avisos.exists():
        return pd.read_parquet(r.avisos), json.loads(r.meta.read_text())
    return pd.DataFrame(), {"ultimo_barrido": None, "barridos": []}


def consolidar(fecha: str, raiz: str = "venta") -> dict:
    """Aplica el barrido de `fecha` sobre el estado y escribe el historial del día."""
    r = rutas(raiz)
    previo, meta = cargar_estado(r)
    if meta["ultimo_barrido"] and fecha <= meta["ultimo_barrido"]:
        raise ValueError(f"El {fecha} ya está procesado (último: {meta['ultimo_barrido']}). "
                         "Usa --reconstruir para rehacer desde cero.")

    hoy = pd.read_parquet(ruta_barrido(fecha, raiz))
    hoy["id_inmueble"] = hoy["id_inmueble"].astype(str)
    hoy = hoy.drop_duplicates("id_inmueble").set_index("id_inmueble")
    eventos = []

    if previo.empty:
        estado = hoy.copy()
        estado["primera_vez"] = fecha
        # Primer barrido: no sabemos cuándo apareció cada aviso → todos "descubiertos"
        eventos.append(_evento(fecha, estado.index, "descubierto", despues=estado["precio"].to_numpy()))
        ausencias_contadas, cobertura = False, None
    else:
        previo = previo.set_index("id_inmueble")
        vistos = hoy.index
        nuevos = vistos.difference(previo.index)
        comunes = vistos.intersection(previo.index)
        faltan = previo.index.difference(vistos)

        # Republicados: avisos nuevos que son uno que hoy faltó (o se retiró hace poco) con otro id
        p = previo.loc[faltan]
        desde = (pd.Timestamp(fecha) - pd.Timedelta(days=REPUBLICADO_DIAS)).strftime("%Y-%m-%d")
        puede = p["estado_aviso"].eq("activo") | (p["fecha_baja"].notna() & (p["fecha_baja"].astype(str) >= desde))
        if "republicado_como" in p:
            puede &= p["republicado_como"].isna()
        rep = republicaciones(p[puede], hoy.loc[nuevos])
        nuevos = nuevos.difference(rep["nuevo"])
        eventos.append(_evento(fecha, rep["nuevo"], "republicado",
                               antes=pd.to_numeric(previo.loc[rep["viejo"], "precio"], errors="coerce").to_numpy(),
                               despues=pd.to_numeric(hoy.loc[rep["nuevo"], "precio"], errors="coerce").to_numpy(),
                               texto_antes=rep["viejo"].tolist()))

        # Altas (publicadas desde el barrido anterior), descubiertos (ya existían) y reapariciones
        publicado = hoy.loc[nuevos, "fecha_publicacion"].fillna("").astype(str)
        es_alta = (publicado >= meta["ultimo_barrido"]).to_numpy()
        if hoy["fecha_publicacion"].isna().all():      # el portal no la publica (Metrocuadrado)
            es_alta = np.ones(len(nuevos), dtype=bool)
        for ids, nombre in ((nuevos[es_alta], "alta"), (nuevos[~es_alta], "descubierto")):
            eventos.append(_evento(fecha, ids, nombre, despues=hoy.loc[ids, "precio"].to_numpy()))
        volvieron = comunes[previo.loc[comunes, "estado_aviso"].eq("retirado").to_numpy()]
        de_barrido = previo.loc[volvieron, "fecha_baja"].notna().to_numpy()     # sin fecha: del rescate
        for ids, nombre in ((volvieron[de_barrido], "reaparece"), (volvieron[~de_barrido], "reactivado")):
            eventos.append(_evento(fecha, ids, nombre, despues=hoy.loc[ids, "precio"].to_numpy()))

        # Cambios en los campos vigilados (subidas y bajadas)
        for campo, nombre in VIGILADOS.items():
            a = pd.to_numeric(previo.loc[comunes, campo], errors="coerce")
            d = pd.to_numeric(hoy.loc[comunes, campo], errors="coerce")
            cambio = a.notna() & d.notna() & (a != d)
            ids = comunes[cambio.to_numpy()]
            eventos.append(_evento(fecha, ids, nombre, antes=a[cambio].to_numpy(), despues=d[cambio].to_numpy()))
        da, dd = _texto(previo.loc[comunes, "descripcion"]), _texto(hoy.loc[comunes, "descripcion"])
        cambio = (da != dd) & da.ne("") & dd.ne("")
        ids = comunes[cambio.to_numpy()]
        eventos.append(_evento(fecha, ids, "cambio_descripcion",
                               texto_antes=previo.loc[ids, "descripcion"].tolist()))

        # Estado nuevo: lo de hoy pisa los datos; los que faltan conservan su última versión
        activos_previos = previo["estado_aviso"].eq("activo").sum()
        cobertura = _cobertura(fecha, raiz)
        ausencias_contadas = (len(vistos) >= MIN_COBERTURA_RELATIVA * activos_previos if cobertura is None else
                              cobertura >= MIN_COBERTURA_PORTAL and len(vistos) >= PISO_RELATIVO * activos_previos)
        estado = pd.concat([hoy, previo.loc[faltan, hoy.columns.intersection(previo.columns)]])
        estado["primera_vez"] = previo["primera_vez"].reindex(estado.index).fillna(fecha)
        estado.loc[rep["nuevo"], "primera_vez"] = previo.loc[rep["viejo"], "primera_vez"].to_numpy()   # hereda
        # Columnas que no vienen en el barrido (p. ej. `origen` de lo rescatado): se conservan
        for c in previo.columns.difference(hoy.columns).difference(CONTROL):
            estado[c] = previo[c].reindex(estado.index)
        for c in ("ultima_vez", "estado_aviso", "fecha_baja", "ausencias", "primera_ausencia"):
            estado.loc[faltan, c] = previo.loc[faltan, c]

        f = faltan[previo.loc[faltan, "estado_aviso"].eq("activo").to_numpy()]      # activos que faltaron hoy
        f = f.difference(rep["viejo"])
        if len(rep):                    # el viejo se retira ya, sin baja, apuntando al id nuevo
            v = pd.Index(rep["viejo"])
            if "republicado_como" not in estado:
                estado["republicado_como"] = None
            estado.loc[v, "republicado_como"] = rep["nuevo"].to_numpy()
            activo = v[previo.loc[v, "estado_aviso"].eq("activo").to_numpy()]
            estado.loc[activo, "fecha_baja"] = _primer_dia_sin_ver(previo.loc[activo, "ultima_vez"],
                                                                   meta["barridos"] + [fecha])
            estado.loc[activo, "estado_aviso"] = "retirado"
        if ausencias_contadas and len(f):
            estado.loc[f, "ausencias"] = previo.loc[f, "ausencias"].fillna(0) + 1
            primera = estado.loc[f, "ausencias"].eq(1)
            estado.loc[f[primera.to_numpy()], "primera_ausencia"] = fecha
            baja = f[estado.loc[f, "ausencias"].ge(AUSENCIAS_PARA_BAJA).to_numpy()]
            estado.loc[baja, "estado_aviso"] = "retirado"
            estado.loc[baja, "fecha_baja"] = estado.loc[baja, "primera_ausencia"]
            eventos.append(_evento(fecha, baja, "baja", antes=pd.to_numeric(estado.loc[baja, "precio"]).to_numpy()))

    # Los vistos hoy quedan activos y con el contador en cero
    vistos = hoy.index
    estado.loc[vistos, "ultima_vez"] = fecha
    estado.loc[vistos, "estado_aviso"] = "activo"
    estado.loc[vistos, "fecha_baja"] = None
    estado.loc[vistos, "ausencias"] = 0
    estado.loc[vistos, "primera_ausencia"] = None

    con_datos = [e for e in eventos if len(e)]
    hist = pd.concat(con_datos, ignore_index=True) if con_datos else _evento(fecha, [], "sin_cambios")
    r.historial.mkdir(parents=True, exist_ok=True)
    from src.almacen import COMPRESION
    guardar_parquet(estado.reset_index().rename(columns={"index": "id_inmueble"}), r.avisos, compression=COMPRESION)
    guardar_parquet(hist, r.historial / f"fecha={fecha}.parquet", compression=COMPRESION)

    cp = hist[hist["evento"] == "cambio_precio"]
    resumen = {"raiz": raiz, "fecha": fecha, "vistos": len(vistos), "avisos_total": len(estado),
               "activos": int(estado["estado_aviso"].eq("activo").sum()),
               "retirados": int(estado["estado_aviso"].eq("retirado").sum()),
               "ausencias_contadas": bool(ausencias_contadas), "cobertura_portal": cobertura,
               "precio_baja": int((cp["valor_despues"] < cp["valor_antes"]).sum()),
               "precio_sube": int((cp["valor_despues"] > cp["valor_antes"]).sum()),
               **hist["evento"].value_counts().to_dict()}
    # El monitor (src/monitor.py) lee el resumen del día y el acumulado: la tasa de "reaparece"
    # sobre las bajas dice si la regla de 3 ausencias está bien calibrada
    acumulado = meta.get("acumulado", {})
    for ev in ("alta", "baja", "reaparece", "republicado"):
        acumulado[ev] = acumulado.get(ev, 0) + int(resumen.get(ev, 0))
    meta = meta | {"ultimo_barrido": fecha, "barridos": meta["barridos"] + [fecha],
                   "ultimo_resumen": resumen, "acumulado": acumulado}
    r.meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, ensure_ascii=False))
    return resumen


def subir(fecha: str, raiz: str = "venta") -> None:
    """avisos (+ respaldo diario y, los lunes, semanal), historial del día y meta a R2."""
    from src import almacen
    r = rutas(raiz)
    almacen.subir_archivo(r.avisos, f"{r.r2}/avisos.parquet")
    almacen.subir_archivo(r.meta, f"{r.r2}/meta.json")
    almacen.subir_archivo(r.historial / f"fecha={fecha}.parquet", f"{r.r2}/historial/fecha={fecha}.parquet")
    almacen.subir_archivo(r.avisos, f"respaldos/diario/fecha={fecha}/{r.respaldo}")
    if pd.Timestamp(fecha).dayofweek == 0:
        almacen.subir_archivo(r.avisos, f"respaldos/semanal/fecha={fecha}/{r.respaldo}")


def bajar(raiz: str = "venta") -> None:
    """Trae de R2 el estado actual (si existe) para continuar desde ahí."""
    from src import almacen
    r = rutas(raiz)
    destino = BASE_DIR / "data" / "_r2tmp"
    almacen.traer(f"{r.r2}/avisos.parquet", destino)
    almacen.traer(f"{r.r2}/meta.json", destino)
    tmp = destino / r.r2
    if (tmp / "avisos.parquet").exists():
        r.estado.mkdir(parents=True, exist_ok=True)
        (tmp / "avisos.parquet").replace(r.avisos)
        (tmp / "meta.json").replace(r.meta)


def reconstruir(raiz: str = "venta") -> None:
    """Rehace avisos + historial desde cero con todos los barridos que haya en disco."""
    r = rutas(raiz)
    for p in [r.avisos, r.meta, *r.historial.glob("*.parquet")]:
        p.unlink(missing_ok=True)
    for fecha in fechas_locales(raiz):
        consolidar(fecha, raiz)


def corregir_republicados(desde: str, raiz: str = "venta", bajar_r2: bool = False, subir_r2: bool = False) -> dict:
    """Corrección única (BACKLOG E2): aplica la regla de republicados a lo consolidado desde `desde` con el
    código anterior (el 30-oct, ~9.000 avisos de una inmobiliaria con id nuevo). Las altas de los ids nuevos
    pasan a `republicado`, las bajas de los viejos salen del historial y el estado queda como si la regla
    hubiera estado. Además, los activos que no se ven desde `desde` reciben las ausencias que la protección
    no les contó (en arriendo saltó por la republicación): el próximo barrido les da la baja con su fecha
    real. Antes copia lo que toca a <estado>/respaldo_republicados/ (y a R2 con `subir_r2`)."""
    import shutil
    from src import almacen
    r = rutas(raiz)
    if bajar_r2:
        bajar(raiz)
        tmp = BASE_DIR / "data" / "_r2tmp"
        almacen.traer(f"{r.r2}/historial/", tmp)
        r.historial.mkdir(parents=True, exist_ok=True)
        for h in (tmp / r.r2 / "historial").glob("fecha=*.parquet"):
            h.replace(r.historial / h.name)
    estado, meta = cargar_estado(r)
    if any(c["que"] == "republicados" and c["desde"] == desde for c in meta.get("correcciones", [])):
        raise ValueError(f"La corrección de republicados desde el {desde} ya está hecha en {raiz}.")
    e = estado.set_index("id_inmueble")
    antes = (pd.Timestamp(desde) - pd.Timedelta(days=REPUBLICADO_DIAS)).strftime("%Y-%m-%d")
    archivos = [r.historial / f"fecha={d}.parquet" for d in meta["barridos"] if d >= antes]   # bajas de los viejos
    respaldo = r.estado / "respaldo_republicados"
    respaldo.mkdir(exist_ok=True)
    for p in [r.avisos, r.meta, *archivos]:
        shutil.copy(p, respaldo / p.name)

    # Pares, con la misma ventana que consolidar: nuevos desde `desde`, viejos que dejaron de verse en los
    # REPUBLICADO_DIAS anteriores al nuevo
    fuera = pd.Series(_primer_dia_sin_ver(e["ultima_vez"], meta["barridos"]), index=e.index)
    ya = e["republicado_como"].notna() if "republicado_como" in e else pd.Series(False, index=e.index)
    viejos = e[fuera.notna() & (fuera.fillna("") >= antes) & ~ya]
    nuevos = e[(e["primera_vez"].astype(str) >= desde) & ~e.index.isin(e.get("republicado_como", pd.Series()).dropna())]
    rep = republicaciones(viejos, nuevos)
    salio = pd.to_datetime(fuera.loc[rep["viejo"]].to_numpy())
    aparecio = pd.to_datetime(e.loc[rep["nuevo"], "primera_vez"].astype(str).to_numpy())
    rep = rep[(aparecio >= salio) & (aparecio <= salio + pd.Timedelta(days=REPUBLICADO_DIAS))]
    v, n = pd.Index(rep["viejo"]), pd.Index(rep["nuevo"])

    e.loc[n, "primera_vez"] = e.loc[v, "primera_vez"].to_numpy()
    if "republicado_como" not in e:
        e["republicado_como"] = None
    e.loc[v, "republicado_como"] = n.to_numpy()
    activo = v[e.loc[v, "estado_aviso"].eq("activo").to_numpy()]
    e.loc[activo, "fecha_baja"] = fuera.loc[activo].to_numpy()
    e.loc[activo, "estado_aviso"] = "retirado"
    # Ausencias que no se contaron (protección): los días de barrido desde que dejó de verse
    sin = e.index[e["estado_aviso"].eq("activo") & fuera.notna() & (fuera.fillna("") >= antes)]
    b = np.array(sorted(meta["barridos"]))
    e.loc[sin, "primera_ausencia"] = fuera.loc[sin].to_numpy()
    e.loc[sin, "ausencias"] = len(b) - np.searchsorted(b, fuera.loc[sin].to_numpy())

    # Historial: altas/descubiertos de los nuevos → republicado; bajas de los viejos, fuera
    precio_viejo = pd.to_numeric(e.loc[v, "precio"], errors="coerce").set_axis(n)
    viejo_de = pd.Series(v, index=n)
    cambios = {"alta": 0, "baja": 0}
    for p in archivos:
        h = pd.read_parquet(p)
        h["id_inmueble"] = h["id_inmueble"].astype(str)
        es_nuevo = h["id_inmueble"].isin(n) & h["evento"].isin(["alta", "descubierto"])
        es_baja = h["id_inmueble"].isin(v) & h["evento"].eq("baja")
        if not (es_nuevo.any() or es_baja.any()):
            continue
        cambios["alta"] += int((es_nuevo & h["evento"].eq("alta")).sum())
        cambios["baja"] += int(es_baja.sum())
        ids = h.loc[es_nuevo, "id_inmueble"]
        h.loc[es_nuevo, "valor_antes"] = precio_viejo.loc[ids].to_numpy()
        h.loc[es_nuevo, "texto_antes"] = viejo_de.loc[ids].to_numpy()
        h.loc[es_nuevo, "evento"] = "republicado"
        h = h[~es_baja]
        guardar_parquet(h, p, compression=almacen.COMPRESION)

    acu = meta.get("acumulado", {})
    acu["alta"] = acu.get("alta", 0) - cambios["alta"]
    acu["baja"] = acu.get("baja", 0) - cambios["baja"]
    acu["republicado"] = acu.get("republicado", 0) + len(rep)
    meta["acumulado"] = acu
    meta.setdefault("correcciones", []).append(
        {"que": "republicados", "desde": desde, "hecha": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
         "pares": len(rep), "altas_a_republicado": cambios["alta"], "bajas_quitadas": cambios["baja"],
         "ausencias_repuestas": len(sin)})
    guardar_parquet(e.reset_index().rename(columns={"index": "id_inmueble"}), r.avisos, compression=almacen.COMPRESION)
    r.meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta["correcciones"][-1], ensure_ascii=False))
    if subir_r2:
        hoy = meta["correcciones"][-1]["hecha"]
        for p in respaldo.iterdir():
            almacen.subir_archivo(p, f"respaldos/correcciones/fecha={hoy}/{r.r2}/{p.name}")
        almacen.subir_archivo(r.avisos, f"{r.r2}/avisos.parquet")
        almacen.subir_archivo(r.meta, f"{r.r2}/meta.json")
        for p in archivos:
            almacen.subir_archivo(p, f"{r.r2}/historial/{p.name}")
    return meta["correcciones"][-1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Estado + historia del mercado desde los barridos")
    p.add_argument("--fecha", help="fecha del barrido a aplicar (AAAA-MM-DD)")
    p.add_argument("--reconstruir", action="store_true", help="rehace todo con los barridos locales")
    p.add_argument("--bajar", action="store_true", help="trae el estado actual de R2 antes de aplicar")
    p.add_argument("--subir", action="store_true", help="sube el estado y el historial del día a R2")
    p.add_argument("--raiz", choices=RAICES_ESTADO, default="venta", help="venta, arriendo o m2_venta")
    p.add_argument("--corregir-republicados", metavar="DESDE",
                   help="corrección única: aplica la regla de republicados a lo consolidado desde DESDE")
    a = p.parse_args()
    if a.corregir_republicados:
        corregir_republicados(a.corregir_republicados, a.raiz, a.bajar, a.subir)
    elif a.reconstruir:
        reconstruir(a.raiz)
    else:
        if a.bajar:
            bajar(a.raiz)
        if not a.fecha:
            if not a.bajar:
                p.error("falta --fecha (o --bajar / --reconstruir)")
        else:
            try:
                consolidar(a.fecha, a.raiz)
            except ValueError as e:                  # p. ej. barrido relanzado el mismo día
                print(f"⚠️  {e} No se aplica de nuevo.")
            else:
                if a.subir:
                    subir(a.fecha, a.raiz)
