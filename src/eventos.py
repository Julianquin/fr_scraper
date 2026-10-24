"""
Bitácora de eventos por inmueble: altas y cambios de precio.

El master (data/master/listings.parquet) es una FOTO: una fila por inmueble con su
último estado, así que al cambiar un precio el anterior se pierde. Esta bitácora
(data/master/events.parquet) guarda la PELÍCULA: una fila por cosa que pasó.

    fecha | id_inmueble | evento | precio_antes | precio_despues

    evento: alta · cambio_precio · correccion_precio (precio ×2 o ÷2: arreglo de digitación)

  - `detectar()` + `registrar()` los usa `master.upsert` en cada corrida.
  - `reconstruir()` rehace la bitácora completa comparando las fotos diarias del
    master: las de git (commits del CI de ago a oct de 2026) y, desde la migración,
    las de R2 (respaldos/ en src/almacen.py).
    Las bajas no se pueden reconstruir: hacen falta barridos completos.

Uso:
    python -m src.eventos --reconstruir     # rehace la bitácora desde el historial de git
    python -m src.eventos                   # resumen de la bitácora actual
"""
from __future__ import annotations
import argparse
import io
import re
import subprocess

import numpy as np
import pandas as pd

from src.config import BASE_DIR

EVENTOS_PATH = BASE_DIR / "data" / "master" / "events.parquet"
MASTER_REL = "data/master/listings.parquet"
COLUMNAS = ["fecha", "id_inmueble", "evento", "precio_antes", "precio_despues"]


def a_numero(valor) -> float:
    """'$ 330.000.000' → 330000000.0 (NaN si no hay cifra, p. ej. 'Precio a convenir')."""
    if not isinstance(valor, str):
        return np.nan
    digitos = re.sub(r"\D", "", valor.split("$")[-1])
    return float(digitos) if digitos else np.nan


def _precios(df: pd.DataFrame) -> pd.Series:
    """Serie id_inmueble → precio numérico."""
    return pd.Series(df["Precio listado"].map(a_numero).values,
                     index=df["id_inmueble"].astype(str).values)


def detectar(previo: pd.DataFrame, nuevo: pd.DataFrame,
             fecha_alta: pd.Series | str, fecha_cambio: pd.Series | str) -> pd.DataFrame:
    """Eventos entre dos estados (DataFrames con id_inmueble y 'Precio listado').

    `fecha_alta` / `fecha_cambio`: una fecha fija o una Serie indexada por id_inmueble.
    """
    antes = _precios(previo) if not previo.empty else pd.Series(dtype=float)
    ahora = _precios(nuevo)
    antes = antes[~antes.index.duplicated(keep="last")]
    ahora = ahora[~ahora.index.duplicated(keep="last")]

    def _fecha(f, ids):
        return f.reindex(ids).values if isinstance(f, pd.Series) else f

    ids_alta = ahora.index.difference(antes.index)
    altas = pd.DataFrame({"fecha": _fecha(fecha_alta, ids_alta), "id_inmueble": ids_alta,
                          "evento": "alta", "precio_antes": np.nan,
                          "precio_despues": ahora[ids_alta].values})

    comunes = ahora.index.intersection(antes.index)
    a, d = antes[comunes], ahora[comunes]
    cambio = a.notna() & d.notna() & (a != d)
    ids_cambio = comunes[cambio.values]
    # Más del doble o menos de la mitad = arreglo de digitación ($330.000.000.000 → $330.000.000)
    ratio = d[ids_cambio].values / a[ids_cambio].values
    tipo = np.where((ratio > 0.5) & (ratio < 2), "cambio_precio", "correccion_precio")
    cambios = pd.DataFrame({"fecha": _fecha(fecha_cambio, ids_cambio), "id_inmueble": ids_cambio,
                            "evento": tipo, "precio_antes": a[ids_cambio].values,
                            "precio_despues": d[ids_cambio].values})
    return pd.concat([altas, cambios], ignore_index=True)[COLUMNAS]


def cargar() -> pd.DataFrame:
    if EVENTOS_PATH.exists():
        return pd.read_parquet(EVENTOS_PATH)
    return pd.DataFrame(columns=COLUMNAS)


def _guardar(ev: pd.DataFrame) -> None:
    ev = ev.astype({"fecha": str, "id_inmueble": str, "evento": str,
                    "precio_antes": float, "precio_despues": float})
    ev = ev.drop_duplicates(subset=["id_inmueble", "evento", "fecha", "precio_despues"])
    EVENTOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ev.sort_values(["fecha", "id_inmueble"]).to_parquet(EVENTOS_PATH, index=False)


def registrar(nuevos: pd.DataFrame) -> int:
    """Agrega eventos a la bitácora. Devuelve cuántos se agregaron."""
    if nuevos.empty:
        return 0
    _guardar(pd.concat([cargar(), nuevos], ignore_index=True))
    return len(nuevos)


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(BASE_DIR), *args],
                          check=True, capture_output=True).stdout


def fotos_master(nombre: str = "listings.parquet"):
    """(fecha, etiqueta, contenido en bytes) de cada foto del master, en orden.

    Hasta oct-2026 el CI commiteaba el master a diario, así que las fotos viven en git.
    Desde la migración a R2 (src/almacen.py) están en respaldos/fecha=AAAA-MM-DD/.
    Se usa git hasta su último día y R2 para los días posteriores.
    """
    ruta = f"data/master/{nombre}"
    ultimo_git = ""
    for linea in _git("log", "--reverse", "--format=%H %cs", "--", ruta).decode().splitlines():
        sha, dia = linea.split()
        try:
            contenido = _git("show", f"{sha}:{ruta}")
        except subprocess.CalledProcessError:
            continue                                   # commit que borró el archivo
        ultimo_git = dia
        yield dia, sha[:7], contenido

    from src import almacen
    if not almacen.configurado():
        return
    s3, bucket = almacen._cliente()
    claves = sorted(o["Key"] for pag in s3.get_paginator("list_objects_v2")
                    .paginate(Bucket=bucket, Prefix="respaldos/")
                    for o in pag.get("Contents", []) if o["Key"].endswith(f"/{nombre}"))
    for clave in claves:
        dia = clave.split("fecha=")[1][:10]
        if dia > ultimo_git:
            yield dia, "R2", s3.get_object(Bucket=bucket, Key=clave)["Body"].read()


def reconstruir() -> pd.DataFrame:
    """Rehace la bitácora comparando cada foto del master con la anterior (git + R2)."""
    partes, previo = [], pd.DataFrame()
    for fecha_commit, etiqueta, contenido in fotos_master():
        foto = pd.read_parquet(io.BytesIO(contenido))
        if "id_inmueble" not in foto or "Precio listado" not in foto:
            continue
        foto["id_inmueble"] = foto["id_inmueble"].astype(str)
        foto = foto.drop_duplicates("id_inmueble", keep="last").reset_index(drop=True)
        ids = foto["id_inmueble"]
        foto["_f"] = (foto["fecha_recoleccion"].fillna(fecha_commit).astype(str).str[:10]
                      if "fecha_recoleccion" in foto else fecha_commit)

        # Una versión más vieja que la ya vista es una reinyección (CSV viejos en el CI
        # hasta oct-2026): se ignora y el aviso conserva su precio anterior.
        if not previo.empty:
            p = previo.set_index("id_inmueble")
            f_prev = ids.map(p["_f"])
            vieja = f_prev.notna() & (foto["_f"] < f_prev)
            foto.loc[vieja, "Precio listado"] = ids[vieja].map(p["Precio listado"])
            foto.loc[vieja, "_f"] = f_prev[vieja]

        # Alta: first_seen si existe; cambio: cuándo se recolectó el precio nuevo
        fs = (foto["first_seen"].fillna(fecha_commit).astype(str).str[:10]
              if "first_seen" in foto else pd.Series(fecha_commit, index=foto.index))
        partes.append(detectar(previo, foto, pd.Series(fs.values, index=ids.values),
                               pd.Series(foto["_f"].values, index=ids.values)))
        previo = foto[["id_inmueble", "Precio listado", "_f"]]
        print(f"  {fecha_commit} {etiqueta}: {len(partes[-1]):>7,} eventos")

    ev = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=COLUMNAS)
    if EVENTOS_PATH.exists():
        EVENTOS_PATH.unlink()
    _guardar(ev)
    return cargar()


def resumen(ev: pd.DataFrame) -> None:
    if ev.empty:
        print("Bitácora vacía.")
        return
    print(f"{len(ev):,} eventos · {ev['id_inmueble'].nunique():,} inmuebles · "
          f"{ev['fecha'].min()} → {ev['fecha'].max()}")
    mes = ev.assign(mes=ev["fecha"].str[:7]).pivot_table(
        index="mes", columns="evento", values="id_inmueble", aggfunc="count", fill_value=0)
    print(mes.to_string())
    c = ev[ev["evento"] == "cambio_precio"]
    if not c.empty:
        var = c["precio_despues"] / c["precio_antes"] - 1
        print(f"Cambios de precio: {(var < 0).sum():,} bajadas · {(var > 0).sum():,} subidas · "
              f"mediana de las bajadas {var[var < 0].median():.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reconstruir", action="store_true", help="Rehace la bitácora desde el historial de git")
    args = p.parse_args()
    resumen(reconstruir() if args.reconstruir else cargar())
