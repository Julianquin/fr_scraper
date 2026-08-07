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

MASTER_PATH = BASE_DIR / "data" / "master" / "listings.parquet"

# Columnas pesadas o basura que no necesita ni el dashboard ni la limpieza.
# (Descripción completa = texto largo; el resto se descarta en preprocessing.)
DROP_COLS = {
    "Descripción completa", "URL imagen", "Acción disponible", "Estado",
    "Estrato", "Parqueaderos", "Financiación", "Formas de pago", "Cuota inicial",
    "Pisos interiores", "Aplica subsidio", "Unidades", "Error detalle",
}


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
    if master.empty:
        nuevos["first_seen"] = run_date
        combinado, n_new, n_upd = nuevos, len(nuevos), 0
    else:
        master["id_inmueble"] = master["id_inmueble"].astype(str)
        prev_ids = set(master["id_inmueble"])
        first_seen = dict(zip(master["id_inmueble"],
                              master.get("first_seen", pd.Series(index=master.index, dtype=str))))
        nuevos["first_seen"] = nuevos["id_inmueble"].map(first_seen).fillna(run_date)
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

    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    combinado.to_parquet(MASTER_PATH, index=False)
    return {"nuevos": n_new, "actualizados": n_upd, "total": len(combinado)}
