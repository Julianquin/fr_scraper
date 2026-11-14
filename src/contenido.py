"""
Cifras neutras para las piezas de redes: solo números y nombres de zonas, sin textos, diseño ni marca (el repositorio
es público). Las piezas (vídeos, carruseles, historias) se hacen fuera del repositorio a partir de estas cifras.

Dos productos:
  - `ficha(r)`: una oportunidad en cifras. canal.py la guarda con cada propuesta en canal/estado.json y pasa a
    «publicados» cuando se aprueba: de ahí sale el vídeo de la oportunidad del día.
  - `semana(ciudad)`: el mercado de los últimos 7 días (pulso, precio por m² y rentabilidad por comuna y barrio,
    mayores bajadas, captación en agregado). canal.yml lo deja cada día, tras proponer, en
        contenido/semana.json              la última semana (siempre la misma clave)
        contenido/semana/<hasta>.json      copia con la fecha del último barrido
    Sin claves de R2 (pruebas en local), en data/contenido/.

Uso:
    python -m src.contenido semana [--ciudad Cali] [--no-subir]    # --no-subir: imprime el resumen y no guarda
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from src.config import BASE_DIR

LOCAL = BASE_DIR / "data" / "contenido"
COMPARABLES = 3                 # similares en la ficha (los mismos que enseña el canal)


def _num(x, dec: int = 0):
    """Número limpio para JSON: None si falta; entero si dec=0."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or (not isinstance(x, str) and pd.isna(x)):
        return None
    return int(round(float(x))) if dec == 0 else round(float(x), dec)


def _fecha(x) -> str | None:
    return None if x is None or pd.isna(x) else f"{pd.Timestamp(x):%Y-%m-%d}"


def _texto(x) -> str | None:
    return None if x is None or (not isinstance(x, str) and pd.isna(x)) else str(x)


def _limpio(v):
    """Lo que json no sabe escribir (numpy, Timestamp, NaN) a tipos de Python."""
    if isinstance(v, dict):
        return {str(k): _limpio(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_limpio(x) for x in v]
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return _fecha(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ── La oportunidad en cifras ──────────────────────────────────────────────────────────────────────────────
def ficha(r: pd.Series) -> dict:
    """Una oportunidad (o una bajada) en cifras: lo que hace falta para contarla en un vídeo o una imagen.
    `r`: una fila de oportunidades.enriquecer (con similares y modelo). Lo que falte queda en None."""
    g = r.get
    comps = g("comps")
    comparables = [] if not isinstance(comps, pd.DataFrame) else [
        {"precio": _num(c.get("Precio")), "area_m2": _num(c.get("Area_m2")), "precio_m2": _num(c.get("precio_m2")),
         "habitaciones": _num(c.get("Habitaciones")), "barrio": _texto(c.get("barrio_oficial")) or _texto(c.get("Barrio")),
         "url": _texto(c.get("URL detalle"))}
        for _, c in comps.head(COMPARABLES).iterrows()]
    bajada = None
    if _num(g("baja"), 3):
        bajada = {"pct": _num(g("baja"), 3), "precio_antes": _num(g("precio_antes")),
                  "fecha": _fecha(g("fecha_baja")), "al_republicar": bool(g("al_republicar") is True)}
    return {
        "id": _texto(g("id_inmueble")), "url": _texto(g("URL detalle")), "fuente": _texto(g("fuente")),
        "ciudad": _texto(g("Ciudad")), "tipo": _texto(g("Tipo_propiedad")),
        "area_m2": _num(g("Area_m2")), "habitaciones": _num(g("Habitaciones")), "banos": _num(g("Baños")),
        "parqueaderos": _num(g("Parqueaderos")), "estrato": _num(g("Estrato")),
        "barrio": _texto(g("barrio_oficial")) or _texto(g("Barrio")), "barrio_aviso": _texto(g("Barrio")),
        "comuna": _texto(g("comuna")), "zona": _texto(g("zona_nombre")),
        "lat": _num(g("Latitud"), 5), "lon": _num(g("Longitud"), 5),
        "publicada": _fecha(g("first_seen")),
        "precio": _num(g("Precio")), "precio_m2": _num(g("precio_m2")),
        "similares": {"n": _num(g("n_comps")), "nivel": _texto(g("nivel_comps")),
                      "mediana_m2": _num(g("med_comps")), "p25_m2": _num(g("p25_comps")), "p75_m2": _num(g("p75_comps")),
                      "descuento": _num(g("descuento"), 3)},
        "modelo": {"estimado": _num(g("modelo_precio")), "p10": _num(g("modelo_p10")), "p90": _num(g("modelo_p90")),
                   "bajo": _num(g("bajo_modelo"), 3)},
        "bajada": bajada,
        "comparables": comparables,
    }


# ── El mercado de la semana ───────────────────────────────────────────────────────────────────────────────
def semana(ciudad: str = "Cali") -> dict:
    """El mercado de los últimos 7 días en cifras, del mismo cálculo que el informe semanal."""
    from src import informe_semanal as s
    d = s.datos(ciudad)
    nombre = lambda c: d["nombres"].get(c, c)
    t = d["todas"]
    comunas = [] if t is None or not len(t) else [
        {"codigo": str(c), "nombre": nombre(c), "viviendas": _num(x["viviendas"]), "precio_m2": _num(x["precio_m2"]),
         "dias_publicadas": _num(x["dias_pub"]), "arriendos": _num(x["arriendos"]), "canon_m2": _num(x["canon_m2"]),
         "rentabilidad": _num(x["rentab"], 4), "barrios": x["barrios"]}
        for c, x in t.iterrows()]
    barrios = [{"comuna": str(c), "barrio": b, "precio_m2": _num(v)} for (c, b), v in d["barrios"].items()]
    cc = d["captar_comunas"]
    captacion = {
        "dias": list(s.CAPTAR_DIAS), "sobre_similares": list(s.CAPTAR_SOBRE),
        "viejas": d["viejas"], "piden_de_mas": d["captar_n"],
        "por_comuna": [{"codigo": str(c), "nombre": nombre(c), "viejas": int(x["viejas"]), "piden_de_mas": int(x["sobre"])}
                       for c, x in cc.iterrows()],
    }
    sel_bj = d["sel_bj"]
    bajadas = [ficha(r) for _, r in sel_bj.iterrows()] if len(sel_bj) else []
    return _limpio({
        "ciudad": ciudad, "desde": d["desde"], "hasta": d["hasta"], "generado": pd.Timestamp.now(tz="America/Bogota"),
        "mercado": {
            "viviendas": d["viviendas"], "avisos_fincaraiz": d["avisos_fr"], "avisos_metrocuadrado": d["avisos_m2"],
            "en_los_dos": d["en_los_dos"], "publicadas": d["publicadas"], "retiradas": d["retiradas"],
            "bajadas": d["bajadas"], "bajada_mediana": d["bajada_mediana"], "precio_m2": d["precio_m2"],
            "arriendos": d["arriendos"], "canon_mediano": d["canon_mediano"], "rentabilidad": d["rentab"],
        },
        "oportunidades": {"evaluadas": d["evaluadas"], "bajo_similares_20": d["bajo_umbral"]},
        "comunas": comunas, "barrios": barrios, "bajadas": bajadas, "captacion": captacion,
    })


def guardar(clave: str, valor: dict) -> None:
    texto = json.dumps(valor, ensure_ascii=False, indent=1)
    if not all(os.environ.get(v) for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")):
        p = LOCAL / clave.removeprefix("contenido/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texto)
        print(f"   (sin R2) {p}")
        return
    from src import almacen
    s3, bucket = almacen._cliente()
    s3.put_object(Bucket=bucket, Key=clave, Body=texto.encode(), ContentType="application/json")
    print(f"   R2: {clave}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["semana"])
    p.add_argument("--ciudad", default="Cali")
    p.add_argument("--no-subir", action="store_true")
    a = p.parse_args()
    s = semana(a.ciudad)
    m = s["mercado"]
    print(f"{a.ciudad} {s['desde']} → {s['hasta']}: {m['viviendas']} viviendas · {m['publicadas']} publicadas · "
          f"{m['bajadas']} bajadas · {len(s['comunas'])} comunas · {len(s['bajadas'])} bajadas destacadas · "
          f"captación {s['captacion']['piden_de_mas']}/{s['captacion']['viejas']}")
    if not a.no_subir:
        guardar("contenido/semana.json", s)
        guardar(f"contenido/semana/{s['hasta']}.json", s)


if __name__ == "__main__":
    main()
