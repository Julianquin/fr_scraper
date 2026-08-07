"""
Genera el dataset curado y liviano que consume el dashboard
(data/app/housing_clean.parquet) a partir del parquet limpio de `ingest`.

Uso:
    python -m src.ingest            # produce data/processed/housing_clean.parquet
    python -m src.build_app_dataset # produce data/app/housing_clean.parquet
"""
from pathlib import Path
import pandas as pd
from src.config import DATA_PROC, BASE_DIR
from src.master import MASTER_PATH
from src.preprocessing import preprocesar_datos_finca_raiz

# Solo las columnas que el dashboard necesita (mantiene el parquet pequeño)
APP_COLS = ["id_inmueble", "Título", "URL detalle", "Precio", "Area_m2",
            "Habitaciones", "Baños", "Tipo_propiedad", "Ciudad",
            "Departamento", "Barrio", "Latitud", "Longitud"]


def run() -> None:
    # Prioridad: master incremental (crudo → se limpia) > parquet ya limpio de ingest
    if MASTER_PATH.exists():
        df = preprocesar_datos_finca_raiz(pd.read_parquet(MASTER_PATH))
    else:
        src = DATA_PROC / "housing_clean.parquet"
        if not src.exists():
            raise FileNotFoundError(
                f"No hay master ni {src}. Corre el scraper (--incremental) o `python -m src.ingest`.")
        df = pd.read_parquet(src)

    out = BASE_DIR / "data" / "app" / "housing_clean.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[[c for c in APP_COLS if c in df.columns]].to_parquet(out, index=False)
    print(f"✅ {len(df):,} filas → {out}")


if __name__ == "__main__":
    run()
