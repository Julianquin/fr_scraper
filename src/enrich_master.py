"""
Backfill de atributos del detalle en el master.

Re-consulta la página de detalle de cada inmueble (rápido, en paralelo con
`requests`) y añade la ficha técnica: Estrato, Antigüedad, Parqueaderos, Piso,
Estado, Área construida/privada. Idempotente y **reanudable**: marca cada id como
enriquecido y guarda checkpoints, así puedes cortar y retomar sin perder trabajo.

Uso:
    python -m src.enrich_master                 # todo el master
    python -m src.enrich_master --limit 3000    # prueba (p. ej. una tanda)
    python -m src.enrich_master --workers 12    # más/menos hilos
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import time

import pandas as pd
from bs4 import BeautifulSoup

from src import master as M
from src.scraper import make_session, _parse_detail

CAMPOS = ["Estrato", "Antiguedad", "Parqueaderos", "Piso", "Estado",
          "Area_construida", "Area_privada"]


def _fetch(session, url: str):
    """Devuelve dict de campos (posiblemente vacío) o None si falló la descarga."""
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        d = _parse_detail(BeautifulSoup(r.text, "html.parser"))
        return {k: d.get(k) for k in CAMPOS if d.get(k) is not None}
    except Exception:
        return None


def _guardar(df: pd.DataFrame) -> None:
    """Persiste el master (coerción de columnas object para que parquet no falle)."""
    out = df.reset_index()
    for c in out.select_dtypes(include="object").columns:
        na = out[c].notna()
        out.loc[na, c] = out.loc[na, c].astype(str)
    out.to_parquet(M.MASTER_PATH, index=False)


def run(workers: int = 12, limit: int | None = None, guardar_cada: int = 2000) -> None:
    m = M.cargar()
    if m.empty:
        raise SystemExit("El master está vacío. Corre el scraper primero.")
    for c in CAMPOS + ["_enriquecido"]:
        if c not in m.columns:
            m[c] = pd.NA
    m = m.set_index("id_inmueble")

    # "1" = ya intentado (NA-safe: los NA/"" se consideran pendientes)
    pend = m[~m["_enriquecido"].astype(str).isin(["1", "True"])]
    pend = pend[pend["URL detalle"].notna()]
    if limit:
        pend = pend.head(limit)
    total = len(pend)
    print(f"🔧 A enriquecer: {total:,} de {len(m):,} inmuebles (workers={workers})")
    if total == 0:
        return

    session = make_session(workers)
    t0 = time.time()
    hechos = con_datos = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(_fetch, session, u): idx
               for idx, u in pend["URL detalle"].items()}
        for f in cf.as_completed(fut):
            idx = fut[f]
            res = f.result()
            if res is not None:                                # página bajada
                m.at[idx, "_enriquecido"] = "1"
                if res:
                    con_datos += 1
                    for k, v in res.items():
                        m.at[idx, k] = v
            hechos += 1
            if hechos % guardar_cada == 0:
                _guardar(m)
                rate = hechos / (time.time() - t0)
                eta = (total - hechos) / rate / 60
                print(f"  {hechos:,}/{total:,}  ({con_datos:,} con datos)  "
                      f"{rate:.0f}/s  ETA ~{eta:.0f} min")

    _guardar(m)
    print(f"✅ Listo: {hechos:,} procesados, {con_datos:,} con ficha técnica "
          f"en {(time.time()-t0)/60:.1f} min → {M.MASTER_PATH}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--guardar-cada", type=int, default=2000)
    a = p.parse_args()
    run(workers=a.workers, limit=a.limit, guardar_cada=a.guardar_cada)


if __name__ == "__main__":
    main()
