"""
Estado + historia del mercado a partir de los barridos diarios (bloques E1/E2; arriendos en E3).

Cada barrido lee ~170.000 avisos, pero de un día a otro casi todo es igual. En vez de guardar una
copia diaria, se mantiene:

  avisos     estado/avisos.parquet                 1 fila por aviso que haya existido alguna vez:
                                                   datos más recientes + primera_vez, ultima_vez,
                                                   estado_aviso (activo | retirado), fecha_baja
  historial  estado/historial/fecha=AAAA-MM-DD.parquet   solo lo que cambió ese día (nunca se reescribe):
                                                   alta · descubierto · cambio_precio ·
                                                   cambio_administracion · cambio_descripcion ·
                                                   baja · reaparece

Reglas:
  - Alta vs. descubierto: cada barrido ve ~93% del país y el ~7% que falta cambia de un día a otro.
    Un aviso que vemos por primera vez es "alta" solo si se publicó desde el barrido anterior; si
    ya estaba publicado, es "descubierto" (el 26-oct fueron 588 altas y 4.212 descubiertos).
  - Baja: el aviso falta en 3 barridos seguidos (con 2 habría muchas bajas falsas por ese ~7%
    variable). La fecha de baja es la del primer barrido en que faltó. Las "reaparece" después de
    una baja miden cuántas bajas fueron falsas.
  - Protección: si el barrido trae menos del 90% de los avisos activos, no se cuentan ausencias ese
    día (un grupo caído no debe generar miles de bajas falsas).
  - Cada fecha se procesa una sola vez y en orden.

Arriendos (`--raiz arriendo`): mismas reglas, con su propio estado (data/estado_arriendo/ y
estado_arriendo/ en R2). En arriendos, `precio` es el canon mensual.

Uso:
    python -m src.consolidar --fecha 2026-10-27 [--subir]   # un barrido (el CI, tras juntar)
    python -m src.consolidar --bajar                         # solo trae el estado de R2
    python -m src.consolidar --reconstruir                   # rehace todo con los barridos locales
    python -m src.consolidar --raiz arriendo …               # lo mismo para arriendos
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
MIN_COBERTURA_RELATIVA = 0.90
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
        ausencias_contadas = False
    else:
        previo = previo.set_index("id_inmueble")
        vistos = hoy.index
        nuevos = vistos.difference(previo.index)
        comunes = vistos.intersection(previo.index)

        # Altas (publicadas desde el barrido anterior), descubiertos (ya existían) y reapariciones
        publicado = hoy.loc[nuevos, "fecha_publicacion"].fillna("").astype(str)
        es_alta = (publicado >= meta["ultimo_barrido"]).to_numpy()
        for ids, nombre in ((nuevos[es_alta], "alta"), (nuevos[~es_alta], "descubierto")):
            eventos.append(_evento(fecha, ids, nombre, despues=hoy.loc[ids, "precio"].to_numpy()))
        volvieron = comunes[previo.loc[comunes, "estado_aviso"].eq("retirado").to_numpy()]
        eventos.append(_evento(fecha, volvieron, "reaparece", despues=hoy.loc[volvieron, "precio"].to_numpy()))

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
        ausencias_contadas = len(vistos) >= MIN_COBERTURA_RELATIVA * activos_previos
        faltan = previo.index.difference(vistos)
        estado = pd.concat([hoy, previo.loc[faltan, hoy.columns.intersection(previo.columns)]])
        estado["primera_vez"] = previo["primera_vez"].reindex(estado.index).fillna(fecha)
        # Columnas que no vienen en el barrido (p. ej. `origen` de lo rescatado): se conservan
        for c in previo.columns.difference(hoy.columns).difference(CONTROL):
            estado[c] = previo[c].reindex(estado.index)
        for c in ("ultima_vez", "estado_aviso", "fecha_baja", "ausencias", "primera_ausencia"):
            estado.loc[faltan, c] = previo.loc[faltan, c]

        f = faltan[previo.loc[faltan, "estado_aviso"].eq("activo").to_numpy()]      # activos que faltaron hoy
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
    meta = meta | {"ultimo_barrido": fecha, "barridos": meta["barridos"] + [fecha]}
    r.meta.write_text(json.dumps(meta, indent=2))

    resumen = {"raiz": raiz, "fecha": fecha, "vistos": len(vistos), "avisos_total": len(estado),
               "activos": int(estado["estado_aviso"].eq("activo").sum()),
               "retirados": int(estado["estado_aviso"].eq("retirado").sum()),
               "ausencias_contadas": bool(ausencias_contadas),
               **hist["evento"].value_counts().to_dict()}
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Estado + historia del mercado desde los barridos")
    p.add_argument("--fecha", help="fecha del barrido a aplicar (AAAA-MM-DD)")
    p.add_argument("--reconstruir", action="store_true", help="rehace todo con los barridos locales")
    p.add_argument("--bajar", action="store_true", help="trae el estado actual de R2 antes de aplicar")
    p.add_argument("--subir", action="store_true", help="sube el estado y el historial del día a R2")
    p.add_argument("--raiz", choices=RAICES, default="venta", help="venta o arriendo")
    a = p.parse_args()
    if a.reconstruir:
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
