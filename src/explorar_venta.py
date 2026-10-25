"""
Exploración de FincaRaíz por HTTP simple (bloque E1). SOLO LECTURA: no toca el master ni R2.

Hallazgos que motivan este módulo (25-oct-2026):
  - Las páginas de listado traen en <script id="__NEXT_DATA__"> los datos ESTRUCTURADOS de
    sus 21 avisos (precio, administración, fecha de publicación, habitaciones, baños, áreas,
    estrato, amenidades, ubicación jerárquica…). No hace falta navegador ni visitar el detalle.
  - El buscador solo deja recorrer los primeros 10.000 resultados de cada consulta (página 476).
    Desde la 477 repite siempre la misma página. Para barrer todo hay que segmentar.

Acciones (resultados en data/analisis/e1/, que no se versiona):
    python -m src.explorar_venta ventana                 # confirma dónde se congela la paginación
    python -m src.explorar_venta segmentos               # total de avisos por ciudad (urls/*.txt)
    python -m src.explorar_venta muestra --paginas 30    # recientes: tiempo, campos, cobertura
"""
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import BASE_DIR

BASE = "https://www.fincaraiz.com.co"
ORDEN_RECIENTE = "ordenListado=3"
POR_PAGINA = 21
MAX_RESULTADOS = 10_000          # tope del buscador por consulta
OUT_DIR = BASE_DIR / "data" / "analisis" / "e1"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
_NEXT = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def url_pagina(ruta: str, n: int) -> str:
    """ruta = 'venta' o 'venta/cali/valle-del-cauca'."""
    return f"{BASE}/{ruta.strip('/')}" + (f"/pagina{n}" if n > 1 else "") + f"?&{ORDEN_RECIENTE}"


def pedir(sesion: requests.Session, ruta: str, n: int) -> tuple[list[dict], dict, float]:
    """(avisos crudos del JSON, paginatorInfo, segundos) de la página n de `ruta`."""
    t = time.time()
    r = sesion.get(url_pagina(ruta, n), headers=UA, timeout=30)
    r.raise_for_status()
    m = _NEXT.search(r.text)
    if not m:
        raise ValueError(f"Sin __NEXT_DATA__ en {r.url}")
    sf = json.loads(m.group(1))["props"]["pageProps"]["fetchResult"]["searchFast"]
    return sf.get("data") or [], sf.get("paginatorInfo") or {}, time.time() - t


def _nombres(loc: dict, nivel: str) -> str | None:
    vals = loc.get(nivel) or []
    return ", ".join(v.get("name", "") for v in vals) or None


def aplanar(a: dict) -> dict:
    """Un aviso del JSON → una fila plana con los campos útiles."""
    loc = a.get("locations") or {}
    precio = a.get("price") or {}
    admin = a.get("commonExpenses") or {}
    ficha = {f.get("field"): f.get("value") for f in (a.get("technicalSheet") or [])}
    return {
        "id_inmueble": str(a.get("id")),
        "titulo": a.get("title"),
        "url": BASE + (a.get("link") or ""),
        "tipo": (a.get("property_type") or {}).get("name"),
        "operacion": (a.get("operation_type") or {}).get("name"),
        "precio": precio.get("amount"),
        "precio_oculto": precio.get("hidePrice"),
        "moneda": (precio.get("currency") or {}).get("name"),
        "administracion": admin.get("amount"),
        "variacion_precio": json.dumps(a.get("price_variation"), ensure_ascii=False) if a.get("price_variation") else None,
        "descuento": a.get("discount"),
        "fecha_publicacion": a.get("created_at"),
        "fecha_actualizacion": a.get("updated_at"),
        "habitaciones": a.get("bedrooms") if a.get("bedrooms") is not None else a.get("rooms"),
        "banos": a.get("bathrooms"),
        "parqueaderos": a.get("garage"),
        "m2": a.get("m2"),
        "m2_construido": a.get("m2Built"),
        "m2_terreno": a.get("m2Terrain"),
        "estrato": a.get("stratum"),
        "piso": a.get("floor"),
        "antiguedad": a.get("antiquity"),
        "estado": ficha.get("construction_state_name"),
        "latitud": a.get("latitude"),
        "longitud": a.get("longitude"),
        "departamento": _nombres(loc, "state"),
        "ciudad": _nombres(loc, "city"),
        "localidad": _nombres(loc, "locality"),
        "comuna": _nombres(loc, "commune"),
        "zona": _nombres(loc, "zone"),
        "barrio": _nombres(loc, "neighbourhood"),
        "direccion": a.get("address") if a.get("showAddress") else None,
        "anunciante": (a.get("owner") or {}).get("name"),
        "anunciante_id": (a.get("owner") or {}).get("id"),
        "es_proyecto": a.get("isProject"),
        "es_unidad_proyecto": a.get("isProjectUnit"),
        "activo": a.get("active"),
        "vendido": a.get("sold"),
        "amenidades": "|".join(f.get("name", "") for f in (a.get("facilities") or [])) or None,
        "n_imagenes": a.get("image_count"),
        "duplicados": len(a.get("duplicated") or []),
        "codigo_anunciante": a.get("code2"),
        "descripcion": a.get("description"),
    }


def ventana(sesion: requests.Session) -> None:
    """Muestra en qué página deja de cambiar el contenido (tope de 10.000 resultados)."""
    ultima = MAX_RESULTADOS // POR_PAGINA
    for n in (ultima - 1, ultima, ultima + 1, ultima + 100):
        data, pi, dt = pedir(sesion, "venta", n)
        print(f"página {n}: primer id {data[0]['id'] if data else '-'} · "
              f"{data[0]['created_at'] if data else '-'} · total {pi.get('total')} · {dt:.2f}s")
        time.sleep(1)


def rutas_ciudad() -> list[str]:
    """Las rutas ciudad/departamento de urls/*.txt, sin el tipo de inmueble."""
    rutas = set()
    for f in (BASE_DIR / "urls").glob("*.txt"):
        for u in f.read_text().split():
            partes = u.split("fincaraiz.com.co/")[-1].strip("/").split("/")
            if partes[0] == "venta" and len(partes) >= 3:
                rutas.add(f"venta/{partes[-2]}/{partes[-1]}")
    return sorted(rutas)


def segmentos(sesion: requests.Session, pausa: float) -> pd.DataFrame:
    """Total de avisos por ciudad: ¿cuáles pasan de 10.000 y hay que partir más?"""
    filas = []
    for ruta in ["venta"] + rutas_ciudad():
        try:
            _, pi, dt = pedir(sesion, ruta, 1)
            filas.append({"ruta": ruta, "total": pi.get("total"), "paginas": pi.get("lastPage"), "segundos": round(dt, 2)})
        except Exception as e:  # noqa: BLE001
            filas.append({"ruta": ruta, "total": None, "error": str(e)[:80]})
        time.sleep(pausa)
    df = pd.DataFrame(filas).sort_values("total", ascending=False)
    df["supera_10k"] = df["total"] > MAX_RESULTADOS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "segmentos.csv", index=False)
    ciudades = df[df["ruta"] != "venta"]
    print(df.head(15).to_string(index=False))
    print(f"\n{len(ciudades)} ciudades · suman {ciudades['total'].sum():,.0f} avisos "
          f"(el total nacional dice {df.loc[df['ruta'] == 'venta', 'total'].iloc[0]:,.0f}) · "
          f"{ciudades['supera_10k'].sum()} pasan de 10.000")
    return df


def muestra(sesion: requests.Session, paginas: int, pausa: float) -> pd.DataFrame:
    """Recorre las primeras `paginas` de /venta (más recientes) y mide tiempo, campos y cobertura."""
    filas, tiempos = [], []
    for n in range(1, paginas + 1):
        data, _, dt = pedir(sesion, "venta", n)
        tiempos.append(dt)
        filas += [aplanar(a) | {"pagina": n} for a in data]
        time.sleep(pausa)
    df = pd.DataFrame(filas)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "muestra_venta.parquet", index=False)

    print(f"{paginas} páginas · {len(df):,} avisos · {df['id_inmueble'].nunique():,} únicos · "
          f"{sum(tiempos) / len(tiempos):.2f} s por página (sin contar la pausa de {pausa}s)")
    print(f"Publicados entre {df['fecha_publicacion'].min()} y {df['fecha_publicacion'].max()}")

    print("\nCampos con dato (% de avisos):")
    campos = ["precio", "administracion", "fecha_publicacion", "habitaciones", "banos", "m2_construido",
              "estrato", "piso", "antiguedad", "latitud", "barrio", "comuna", "amenidades", "descripcion"]
    llenos = {c: df[c].notna() & (df[c].astype(str).str.strip() != "") & (df[c].astype(str) != "0") for c in campos}
    print(pd.Series({c: f"{v.mean():.0%}" for c, v in llenos.items()}).to_string())
    print(f"\nProyectos: {df['es_proyecto'].mean():.0%} · unidades de proyecto: {df['es_unidad_proyecto'].mean():.0%} · "
          f"con duplicados declarados: {(df['duplicados'] > 0).mean():.0%}")
    print("Tipos:", df["tipo"].value_counts().head(6).to_dict())

    master = BASE_DIR / "data" / "master" / "listings.parquet"
    if master.exists():
        ids = set(pd.read_parquet(master, columns=["id_inmueble"])["id_inmueble"].astype(str))
        df["en_master"] = df["id_inmueble"].isin(ids)
        print(f"\nCobertura: {df['en_master'].mean():.0%} de estos avisos ya están en el master")
        por_fecha = df.groupby("fecha_publicacion")["en_master"].agg(["size", "mean"]).sort_index(ascending=False)
        print(por_fecha.head(8).rename(columns={"size": "avisos", "mean": "en master"})
              .to_string(formatters={"en master": "{:.0%}".format}))
        fuera = df[~df["en_master"]]
        print("Ciudades de los que faltan:", fuera["ciudad"].value_counts().head(8).to_dict())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Exploración de /venta por HTTP (solo lectura)")
    p.add_argument("--pausa", type=float, default=1.0, help="segundos entre peticiones (cortesía)")
    sub = p.add_subparsers(dest="accion", required=True)
    sub.add_parser("ventana")
    sub.add_parser("segmentos")
    m = sub.add_parser("muestra")
    m.add_argument("--paginas", type=int, default=30)
    a = p.parse_args()
    s = requests.Session()
    {"ventana": lambda: ventana(s), "segmentos": lambda: segmentos(s, a.pausa),
     "muestra": lambda: muestra(s, a.paginas, a.pausa)}[a.accion]()
