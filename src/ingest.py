# src/ingest.py
import pandas as pd, os, joblib
from pathlib import Path
from src.config import DATA_RAW, DATA_PROC          
from src.preprocessing import preprocesar_datos_finca_raiz   # lo movemos a preprocessing.py

def run():
    """Consolida todos los snapshots de data/raw, limpia y genera dos Parquets:

    - housing_history.parquet : toda la historia (una fila por inmueble y fecha).
    - housing_clean.parquet   : estado actual (último snapshot de cada inmueble).
    """
    # rglob → lee tanto los CSV sueltos (lote inicial) como las carpetas data/raw/<fecha>/
    files = sorted(DATA_RAW.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {DATA_RAW}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = preprocesar_datos_finca_raiz(df)

    DATA_PROC.mkdir(parents=True, exist_ok=True)

    # 1) Historia completa: todas las observaciones (id + fecha)
    hist_out = DATA_PROC / "housing_history.parquet"
    df.to_parquet(hist_out, index=False)

    # 2) Estado actual: la observación más reciente de cada inmueble
    latest = (
        df.sort_values("fecha_recoleccion")
          .groupby("id_inmueble", as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )
    clean_out = DATA_PROC / "housing_clean.parquet"
    latest.to_parquet(clean_out, index=False)

    print(f"✅ Historia   → {hist_out} — {len(df):,} filas")
    print(f"✅ Estado actual → {clean_out} — {len(latest):,} inmuebles únicos")

if __name__ == "__main__":
    run()






