"""
Store maestro incremental de inmuebles, keyed por `id_inmueble`.

Guarda un parquet (data/master/listings.parquet) con una fila por inmueble
(los campos crudos del scraper) + `first_seen` / `last_seen`. El scraper lo lee
para saber qué ya conoce (y así cortar la paginación), y el `upsert` lo actualiza
con lo nuevo/cambiado de cada corrida.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pandas as pd
from src.config import BASE_DIR
from src import eventos

MASTER_PATH = BASE_DIR / "data" / "master" / "listings.parquet"

# Columnas pesadas o basura que no necesita ni el dashboard ni la limpieza.
# (Descripción completa = texto largo; el resto se descarta en preprocessing.)
DROP_COLS = {
    "Descripción completa", "URL imagen", "Acción disponible",
    "Financiación", "Formas de pago", "Cuota inicial",
    "Pisos interiores", "Aplica subsidio", "Unidades", "Error detalle",
}
# Nota: Estrato/Parqueaderos/Estado/Antiguedad/Piso ahora SÍ se conservan
# (los captura la ficha técnica del detalle) → son features de alto valor.


def cargar() -> pd.DataFrame:
    """Devuelve el master actual (DataFrame vacío si aún no existe)."""
    if MASTER_PATH.exists():
        return pd.read_parquet(MASTER_PATH)
    return pd.DataFrame()


def precios_conocidos(master: pd.DataFrame | None = None) -> Dict[str, str]:
    """Mapa {id_inmueble: 'Precio listado'} para detectar nuevos/cambios."""
    if master is None:
        master = cargar()
    if master.empty or "id_inmueble" not in master.columns:
        return {}
    return dict(zip(master["id_inmueble"].astype(str),
                    master.get("Precio listado", pd.Series(dtype=str)).astype(str)))


def upsert(rows: List[Dict] | pd.DataFrame, run_date: str) -> Dict[str, int]:
    """Inserta/actualiza filas por `id_inmueble` y persiste el master.

    Devuelve conteos {nuevos, actualizados, total}.
    """
    nuevos = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if nuevos.empty or "id_inmueble" not in nuevos.columns:
        return {"nuevos": 0, "actualizados": 0, "total": len(cargar())}

    nuevos["id_inmueble"] = nuevos["id_inmueble"].astype(str)
    nuevos = nuevos.drop_duplicates(subset="id_inmueble", keep="last")
    nuevos["last_seen"] = run_date

    master = cargar()
    if not master.empty:
        master["id_inmueble"] = master["id_inmueble"].astype(str)
    # Una versión más vieja nunca pisa a una más nueva (pasó con CSV viejos reingestados)
    if not master.empty and "fecha_recoleccion" in nuevos and "fecha_recoleccion" in master:
        actual = master.set_index("id_inmueble")["fecha_recoleccion"].astype(str).str[:10]
        actual = actual[~actual.index.duplicated(keep="last")]
        previa = nuevos["id_inmueble"].map(actual)
        vieja = previa.notna() & (nuevos["fecha_recoleccion"].astype(str).str[:10] < previa)
        nuevos = nuevos[~vieja]
    # Antes de sobrescribir: altas y cambios de precio a la bitácora (src/eventos.py)
    eventos.registrar(eventos.detectar(master, nuevos, run_date, run_date))
    if master.empty:
        nuevos["first_seen"] = run_date
        combinado, n_new, n_upd = nuevos, len(nuevos), 0
    else:
        prev_ids = set(master["id_inmueble"])
        first_seen = dict(zip(master["id_inmueble"],
                              master.get("first_seen", pd.Series(index=master.index, dtype=str))))
        nuevos["first_seen"] = nuevos["id_inmueble"].map(first_seen).fillna(run_date)
        # El listado no trae la ficha técnica del detalle: se conserva la que ya había
        previa = master.drop_duplicates("id_inmueble", keep="last").set_index("id_inmueble")
        for c in [c for c in FICHA if c in previa]:
            conocida = nuevos["id_inmueble"].map(previa[c])
            nuevos[c] = nuevos[c].combine_first(conocida) if c in nuevos else conocida
        # Los registros nuevos ganan; se conservan los del master que no vinieron esta vez
        conservados = master[~master["id_inmueble"].isin(nuevos["id_inmueble"])]
        combinado = pd.concat([conservados, nuevos], ignore_index=True)
        n_new = len(set(nuevos["id_inmueble"]) - prev_ids)
        n_upd = len(nuevos) - n_new

    # Quitar columnas pesadas/basura para mantener el master liviano
    combinado = combinado.drop(columns=[c for c in DROP_COLS if c in combinado.columns])

    # pyarrow no tolera columnas 'object' con tipos mezclados (texto + float NaN),
    # frecuente en los CSV viejos (Financiación, Formas de pago…). Convertimos los
    # valores NO nulos a str para que el parquet tenga un tipo consistente.
    for c in combinado.select_dtypes(include="object").columns:
        notna = combinado[c].notna()
        combinado.loc[notna, c] = combinado.loc[notna, c].astype(str)

    _guardar(combinado)
    return {"nuevos": n_new, "actualizados": n_upd, "total": len(combinado)}


def _guardar(df: pd.DataFrame) -> None:
    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MASTER_PATH, index=False)


# Ficha técnica del detalle: un CSV del listado no la trae, así que al reemplazar una fila
# se perdía. La reparación la recupera de versiones anteriores del mismo aviso.
FICHA = ["Estrato", "Antiguedad", "Parqueaderos", "Piso", "Estado",
         "Area_construida", "Area_privada", "_enriquecido"]


def reparar() -> None:
    """Rehace el master desde sus fotos (git y R2): por aviso, la versión con la
    fecha_recoleccion más reciente (ignora reinyecciones de CSV viejos), la ficha
    técnica más reciente que se conozca y last_seen = última recolección real."""
    import io
    mejor, ficha = pd.DataFrame(), pd.DataFrame()
    for _, _, contenido in eventos.fotos_master():     # git hasta la migración, R2 después
        foto = pd.read_parquet(io.BytesIO(contenido))
        if "id_inmueble" not in foto or "fecha_recoleccion" not in foto:
            continue
        foto["id_inmueble"] = foto["id_inmueble"].astype(str)
        foto["fecha_recoleccion"] = foto["fecha_recoleccion"].astype(str).str[:10]
        # Sin fecha (CSV sembrados en agosto) = más vieja que cualquier fecha.
        # Orden estable: a igual fecha gana la foto más reciente
        foto["_orden"] = foto["fecha_recoleccion"].where(
            foto["fecha_recoleccion"].str.match(r"\d{4}-\d{2}-\d{2}", na=False), "0000-00-00")
        mejor = pd.concat([mejor, foto], ignore_index=True) \
                  .sort_values("_orden", kind="stable") \
                  .drop_duplicates("id_inmueble", keep="last")
        cols = ["id_inmueble"] + [c for c in FICHA if c in foto]
        if len(cols) > 1:
            ficha = pd.concat([ficha, foto[cols]], ignore_index=True).groupby("id_inmueble").last()
            ficha = ficha.reset_index()

    actual = cargar()
    actual["id_inmueble"] = actual["id_inmueble"].astype(str)
    mejor = mejor.set_index("id_inmueble").reindex(actual["id_inmueble"])
    cambiadas = (mejor["Precio listado"].fillna("").values
                 != actual["Precio listado"].fillna("").values).sum()

    rep = mejor.copy()
    rep["first_seen"] = actual["first_seen"].values
    # last_seen = última recolección real; sin fecha, se conserva el que tenía
    con_fecha = rep["_orden"] != "0000-00-00"
    rep["last_seen"] = rep["fecha_recoleccion"].where(con_fecha, actual["last_seen"].values)
    f = ficha.set_index("id_inmueble").reindex(rep.index)
    antes = rep[[c for c in FICHA if c in rep]].notna().sum().sum()
    for c in f.columns:
        rep[c] = rep[c].combine_first(f[c]) if c in rep else f[c]
    recuperados = rep[[c for c in FICHA if c in rep]].notna().sum().sum() - antes

    rep = rep.reset_index()[[c for c in actual.columns if c in rep.columns or c == "id_inmueble"]]
    rep["fecha_recoleccion"] = rep["fecha_recoleccion"].where(con_fecha.values)
    for c in rep.select_dtypes(include=["object", "string"]).columns:
        notna = rep[c].notna()
        rep.loc[notna, c] = rep.loc[notna, c].astype(str)
    _guardar(rep)
    print(f"🛠️  Master reparado: {cambiadas:,} avisos recuperan su precio más reciente · "
          f"{recuperados:,} datos de ficha técnica recuperados · {len(rep):,} avisos")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reparar", action="store_true", help="Rehace el master desde su historial en git")
    if p.parse_args().reparar:
        reparar()
