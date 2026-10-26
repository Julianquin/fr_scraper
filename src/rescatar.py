"""
Rescata al estado nuevo (estado/avisos + estado/historial) lo que solo sabe el sistema viejo,
antes de apagar el scraper de las 570 URLs. Se corre una vez; repetirlo no duplica nada.

Qué se rescata (todo marcado con su origen para poder filtrarlo):
  1. Avisos vistos ago–oct 2026 que el barrido ya no encuentra → en `avisos` como retirados
     (origen = scraper_viejo). Los vistos en la última semana antes del barrido se dejan fuera:
     pueden seguir activos y el barrido los descubrirá.
  2. Avisos de la foto de julio de 2025 que no volvieron a aparecer → retirados (origen = foto_2025).
  3. Cambios de precio de la bitácora vieja anteriores al barrido → `historial` en los archivos de
     sus fechas (fuente = scraper_viejo). No se tocan los días del barrido (desde el 25-oct-2026).
  4. Precio de julio de 2025 de cada aviso → evento `observado` (fuente = foto_2025): permite
     comparar un año después.
  5. `primera_vez` toma la fecha más antigua conocida (julio 2025 o la primera vez del viejo).

No se rescata: datos de proyectos (fuera del producto), el texto de tipología (ya está como campos)
ni el estado "activo" del viejo (estaba inflado: no detectaba bajas).

Uso:
    python -m src.consolidar --bajar --fecha <último>   # (o tener data/estado al día)
    python -m src.rescatar [--subir]
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BASE_DIR, DATA_RAW
from src.consolidar import AVISOS, HISTORIAL, META, _evento
from src.eventos import a_numero
from src.fincaraiz_http import guardar_parquet

INICIO_BARRIDO = "2026-10-25"          # desde aquí manda el barrido
MARGEN_ACTIVOS = "2026-10-18"          # vistos después: pueden seguir activos, no se rescatan
FECHA_2025 = "2025-07-07"
CODIGO_ANTIGUEDAD = {"menor a 1 año": 1, "1 a 8 años": 2, "9 a 15 años": 3, "16 a 30 años": 4,
                     "más de 30 años": 5}


def _equivalencias(viejo: pd.DataFrame, nuevo: pd.DataFrame, col_v: str, col_n: str) -> dict:
    """Nombre viejo → nombre nuevo más común entre los avisos que están en ambos sistemas."""
    j = viejo[["id_inmueble", col_v]].merge(nuevo[["id_inmueble", col_n]], on="id_inmueble").dropna()
    return j.groupby(col_v)[col_n].agg(lambda s: s.value_counts().index[0]).to_dict()


def precios_2025() -> pd.Series:
    """id_inmueble → precio en la foto de julio de 2025 (CSV sueltos de data/raw)."""
    archivos = sorted(Path(DATA_RAW).glob("*.csv"))
    if not archivos:
        return pd.Series(dtype=float)
    d = pd.concat([pd.read_csv(f, usecols=["URL detalle", "Precio listado"], dtype=str) for f in archivos])
    d = d[~d["URL detalle"].str.contains("proyectos-vivienda", na=False)]
    d["id_inmueble"] = d["URL detalle"].str.rstrip("/").str.split("/").str[-1]
    d["precio"] = d["Precio listado"].map(a_numero)
    return d.dropna(subset=["precio"]).drop_duplicates("id_inmueble").set_index("id_inmueble")["precio"]


def rescatar() -> dict:
    from src.build_app_dataset import dataset
    from src import master as M
    from src import eventos as E

    nuevo = pd.read_parquet(AVISOS)
    nuevo["id_inmueble"] = nuevo["id_inmueble"].astype(str)
    if "origen" not in nuevo:
        nuevo["origen"] = "barrido"
    ya = set(nuevo["id_inmueble"])

    raw = M.cargar()[["id_inmueble", "first_seen", "last_seen", "fecha_recoleccion", "Etiquetas"]]
    raw["id_inmueble"] = raw["id_inmueble"].astype(str)
    limpio = dataset(incluir_historico=True)
    limpio["id_inmueble"] = limpio["id_inmueble"].astype(str)
    viejo = limpio.drop(columns=["fecha_recoleccion"]).drop_duplicates("id_inmueble").merge(raw, on="id_inmueble")
    dep = _equivalencias(viejo, nuevo, "Departamento", "departamento")
    ciu = _equivalencias(viejo, nuevo, "Ciudad", "ciudad")
    p25 = precios_2025()

    # ── 1 y 2. Avisos que solo conoce el viejo → retirados ─────────────────────
    solo = viejo[~viejo["id_inmueble"].isin(ya)].copy()
    es_2025 = solo["fecha_recoleccion"].isna()
    recientes = ~es_2025 & (solo["last_seen"].astype(str) >= MARGEN_ACTIVOS)
    solo = solo[~recientes]
    es_2025 = solo["fecha_recoleccion"].isna()
    n = lambda c: pd.to_numeric(solo[c], errors="coerce")
    rescatados = pd.DataFrame({
        "id_inmueble": solo["id_inmueble"],
        "url": solo["URL detalle"],
        "tipo": solo["Tipo_propiedad"].replace("Desconocido", np.nan),
        "precio": n("Precio"),
        "habitaciones": n("Habitaciones"), "banos": n("Baños"), "parqueaderos": n("Parqueaderos"),
        "m2": n("Area_m2"), "estrato": n("Estrato"), "piso": n("Piso"),
        "antiguedad": solo["Antiguedad"].map(CODIGO_ANTIGUEDAD),
        "estado": solo["Estado"],
        "latitud": n("Latitud"), "longitud": n("Longitud"),
        "departamento": solo["Departamento"].map(lambda v: dep.get(v, v)),
        "ciudad": solo["Ciudad"].map(lambda v: ciu.get(v, v)),
        "barrio": solo["Barrio"],
        "anunciante": solo["Publicante"],
        "etiquetas": solo["Etiquetas"].map(lambda e: json.dumps(ast.literal_eval(e), ensure_ascii=False)
                                           if isinstance(e, str) and e.startswith("[") else None),
        "descripcion": solo["Descripción completa"],
        # control: retirado antes del barrido; no sabemos el día exacto de la baja
        "primera_vez": np.where(es_2025, FECHA_2025, solo["first_seen"].astype(str).str[:10]),
        "ultima_vez": np.where(es_2025, FECHA_2025, solo["last_seen"].astype(str).str[:10]),
        "estado_aviso": "retirado",
        "fecha_baja": None,
        "origen": np.where(es_2025, "foto_2025", "scraper_viejo"),
    })
    # Visto en jul-2025 → esa es la primera vez, aunque el viejo lo sembrara el 8-ago
    en_2025 = rescatados["id_inmueble"].isin(p25.index)
    rescatados.loc[en_2025, "primera_vez"] = FECHA_2025

    # ── 5. primera_vez de los que ya están en el estado nuevo ──────────────────
    antes = nuevo["primera_vez"].astype(str)
    fs = nuevo["id_inmueble"].map(raw.set_index("id_inmueble")["first_seen"].astype(str).str[:10])
    fs = fs.where(nuevo["id_inmueble"].map(raw.set_index("id_inmueble")["fecha_recoleccion"]).notna())
    fs = fs.where(~nuevo["id_inmueble"].isin(p25.index), FECHA_2025)
    nuevo["primera_vez"] = pd.concat([antes, fs.fillna(antes)], axis=1).min(axis=1)
    adelantados = int((nuevo["primera_vez"] < antes).sum())

    avisos = pd.concat([nuevo, rescatados], ignore_index=True)
    assert avisos["id_inmueble"].is_unique

    # ── 3. Cambios de precio del viejo, antes del barrido ─────────────────────
    ev = E.cargar()
    ev = ev[ev["evento"].isin(["cambio_precio", "correccion_precio"]) & (ev["fecha"] < INICIO_BARRIDO)]
    ev = ev[ev["id_inmueble"].astype(str).isin(set(avisos["id_inmueble"]))]
    por_fecha = {f: _evento(f, g["id_inmueble"].astype(str), "", antes=g["precio_antes"].to_numpy(),
                            despues=g["precio_despues"].to_numpy(), fuente="scraper_viejo")
                    .assign(evento=g["evento"].to_numpy())
                 for f, g in ev.groupby("fecha")}

    # ── 4. Precio observado en julio de 2025 ───────────────────────────────────
    obs = p25[p25.index.isin(set(avisos["id_inmueble"]))]
    por_fecha[FECHA_2025] = _evento(FECHA_2025, obs.index, "observado", despues=obs.to_numpy(), fuente="foto_2025")

    # ── Guardar ───────────────────────────────────────────────────────────────
    from src.almacen import COMPRESION
    guardar_parquet(avisos, AVISOS, compression=COMPRESION)
    for f, d in por_fecha.items():
        assert f < INICIO_BARRIDO, f"no se reescriben días del barrido ({f})"
        guardar_parquet(d, HISTORIAL / f"fecha={f}.parquet", compression=COMPRESION)
    meta = json.loads(META.read_text())
    resumen = {
        "avisos_antes": len(nuevo), "avisos_despues": len(avisos),
        "retirados_scraper_viejo": int((rescatados["origen"] == "scraper_viejo").sum()),
        "retirados_foto_2025": int((rescatados["origen"] == "foto_2025").sum()),
        "no_rescatados_por_recientes": int(recientes.sum()),
        "primera_vez_adelantada": adelantados,
        "cambios_precio_rescatados": int(sum(len(d) for f, d in por_fecha.items() if f != FECHA_2025)),
        "dias_de_historial_rescatados": len(por_fecha) - 1,
        "precios_observados_2025": len(obs),
    }
    meta["rescate"] = resumen | {"fecha": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")}
    META.write_text(json.dumps(meta, indent=2))
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return {"fechas": sorted(por_fecha)}


def subir(fechas: list[str]) -> None:
    from src import almacen
    almacen.subir_archivo(AVISOS, "estado/avisos.parquet")
    almacen.subir_archivo(META, "estado/meta.json")
    for f in fechas:
        almacen.subir_archivo(HISTORIAL / f"fecha={f}.parquet", f"estado/historial/fecha={f}.parquet")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rescata al estado nuevo lo que solo sabe el sistema viejo")
    p.add_argument("--subir", action="store_true")
    a = p.parse_args()
    r = rescatar()
    if a.subir:
        subir(r["fechas"])
