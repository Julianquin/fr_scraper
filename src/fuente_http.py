"""
Adaptador: el estado del barrido HTTP (estado/avisos + estado/historial, de src/consolidar.py) con
el formato que ya usan la alerta (src/oportunidades.py) y el mapa (src/viz_map.py).

Permite correr ambos con los datos nuevos (`--fuente http`) en paralelo a los del scraper viejo,
para comparar antes del cambio de fuente (cierre de E1). No modifica nada del pipeline actual.

Traducciones (validadas el 26-oct-2026 con los ~86.000 avisos presentes en ambas fuentes):
  - Area_m2      = m2 (coincide con el área del scraper viejo en el 99%)
  - Antiguedad   = código del portal → texto del scraper viejo (coincide en el 99%)
  - first_seen   = fecha_publicacion (fecha real de publicación; antes era "primera vez que lo vimos")
  - Etiquetas    = etiquetas (misma idea: "Usado", "Proyecto", "Precio en dólares"…)
  - cambios de precio del historial → formato de la bitácora (src/eventos.py), con la misma regla:
    ×2 o ÷2 = corrección de digitación
"""
from __future__ import annotations

import pandas as pd

from src.consolidar import AVISOS, HISTORIAL

ANTIGUEDAD = {1: "menor a 1 año", 2: "1 a 8 años", 3: "9 a 15 años", 4: "16 a 30 años", 5: "más de 30 años"}


def dataset_activo() -> pd.DataFrame:
    """Avisos activos en el formato de data/app/housing_clean.parquet (+ first_seen y Etiquetas)."""
    a = pd.read_parquet(AVISOS)
    a = a[a["estado_aviso"] == "activo"]
    n = lambda c: pd.to_numeric(a[c], errors="coerce")
    titulo = a["tipo"].fillna("Inmueble") + " en venta en " + a["barrio"].fillna(a["ciudad"]).fillna("")
    return pd.DataFrame({
        "id_inmueble": a["id_inmueble"].astype(str),
        "Título": titulo,
        "URL detalle": a["url"],
        "Precio": n("precio"),
        "Area_m2": n("m2"),
        "Habitaciones": n("habitaciones"),
        "Baños": n("banos"),
        "Tipo_propiedad": a["tipo"],
        "Ciudad": a["ciudad"],
        "Departamento": a["departamento"],
        "Barrio": a["barrio"],
        "Latitud": n("latitud"),
        "Longitud": n("longitud"),
        "Estrato": n("estrato").where(n("estrato") > 0),
        "Parqueaderos": n("parqueaderos"),
        "Piso": n("piso"),
        "Antiguedad": n("antiguedad").map(ANTIGUEDAD),
        "Estado": a["estado"],                      # estado físico: "Usado", "Excelente estado"…
        "first_seen": pd.to_datetime(a["fecha_publicacion"], errors="coerce"),
        "Etiquetas": a["etiquetas"],
    }).reset_index(drop=True)


def eventos() -> pd.DataFrame:
    """Cambios de precio del historial con el formato de la bitácora (src/eventos.py)."""
    partes = [pd.read_parquet(p) for p in sorted(HISTORIAL.glob("fecha=*.parquet"))]
    h = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(
        columns=["fecha", "id_inmueble", "evento", "valor_antes", "valor_despues"])
    c = h[h["evento"] == "cambio_precio"].rename(columns={"valor_antes": "precio_antes",
                                                          "valor_despues": "precio_despues"})
    ratio = c["precio_despues"] / c["precio_antes"]
    c = c.assign(evento=ratio.between(0.5, 2, inclusive="neither").map(
        {True: "cambio_precio", False: "correccion_precio"}))
    return c[["fecha", "id_inmueble", "evento", "precio_antes", "precio_despues"]].reset_index(drop=True)
