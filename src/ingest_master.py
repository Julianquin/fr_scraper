"""
Upsert de lo scrapeado en el store maestro incremental.

Lee los CSV de la corrida más reciente en data/raw/<fecha>/ y los inserta/actualiza en
data/master/listings.parquet por `id_inmueble`. Pensado para correr una sola vez
en el job `merge` (con los CSV de todos los departamentos ya consolidados), o en
local tras `python -m src.scraper --incremental`.

Uso:
    python -m src.ingest_master
"""
from pathlib import Path
import pandas as pd

from src.config import DATA_RAW
from src import master as M


def run(todas: bool = False) -> None:
    # Solo la corrida más reciente (data/raw/<fecha>/): reingestar carpetas viejas
    # devolvía avisos a precios antiguos. `--todas` para resembrar a propósito.
    carpetas = sorted(p for p in Path(DATA_RAW).glob("*") if p.is_dir())
    if not carpetas:
        print(f"⚠️  No hay corridas en {DATA_RAW}; nada que upsertar.")
        return
    fuentes = carpetas if todas else carpetas[-1:]
    files = sorted(f for c in fuentes for f in c.rglob("*.csv"))
    if not files:
        print(f"⚠️  No hay CSV en {fuentes[-1]}; nada que upsertar.")
        return
    print(f"📥 {len(files)} CSV de {', '.join(c.name for c in fuentes)}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Los CSV viejos no traen id_inmueble → lo derivamos de la URL (último segmento),
    # idéntico a lo que hace el scraper/limpieza. Así se pueden sembrar en el master.
    if "URL detalle" in df.columns:
        df["id_inmueble"] = (
            df["URL detalle"].astype(str).str.rstrip("/").str.split("/").str[-1]
        )

    run_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    stats = M.upsert(df, run_date)
    print(f"🗄️  Master actualizado → {M.MASTER_PATH}")
    print(f"   +{stats['nuevos']} nuevos · {stats['actualizados']} actualizados · "
          f"{stats['total']} inmuebles en total")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--todas", action="store_true", help="Ingesta todas las carpetas de data/raw (resiembra)")
    run(p.parse_args().todas)
