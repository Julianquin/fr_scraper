"""
Bitácora de eventos por inmueble: altas y cambios de precio.

El master (data/master/listings.parquet) es una FOTO: una fila por inmueble con su
último estado, así que al cambiar un precio el anterior se pierde. Esta bitácora
(data/master/events.parquet) guarda la PELÍCULA: una fila por cosa que pasó.

    fecha | id_inmueble | evento | precio_antes | precio_despues

    evento: alta · cambio_precio · correccion_precio (precio ×2 o ÷2: arreglo de digitación)

  - `detectar()` + `registrar()` los usa `master.upsert` en cada corrida.
  - `reconstruir()` rehace la bitácora completa comparando las fotos diarias del
    master guardadas en git (hay una por commit del CI desde agosto de 2026).
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


def reconstruir() -> pd.DataFrame:
    """Rehace la bitácora comparando cada foto del master en git con la anterior."""
    commits = [linea.split() for linea in
               _git("log", "--reverse", "--format=%H %cs", "--", MASTER_REL).decode().splitlines()]
    partes, previo = [], pd.DataFrame()
    for sha, fecha_commit in commits:
        try:
            foto = pd.read_parquet(io.BytesIO(_git("show", f"{sha}:{MASTER_REL}")))
        except subprocess.CalledProcessError:
            continue                                   # commit que borró el archivo
        if "id_inmueble" not in foto or "Precio listado" not in foto:
            continue
        foto["id_inmueble"] = foto["id_inmueble"].astype(str)
        ids = foto["id_inmueble"]

        # Alta: first_seen si existe; cambio: cuándo se recolectó el precio nuevo
        def _col(nombre):
            if nombre not in foto:
                s = pd.Series(fecha_commit, index=ids.values)
            else:
                s = pd.Series(foto[nombre].fillna(fecha_commit).astype(str).str[:10].values,
                              index=ids.values)
            return s[~s.index.duplicated(keep="last")]

        partes.append(detectar(previo, foto, _col("first_seen"), _col("fecha_recoleccion")))
        previo = foto[["id_inmueble", "Precio listado"]]
        print(f"  {fecha_commit} {sha[:7]}: {len(partes[-1]):>7,} eventos")

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
