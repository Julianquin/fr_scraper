"""
Lectura de FincaRaíz por HTTP simple: piezas comunes a la exploración (src/explorar_venta.py)
y al scraper nuevo (src/scraper_http.py).

Las páginas públicas de listado traen en <script id="__NEXT_DATA__"> los datos estructurados de
sus 21 avisos. Se leen como lo haría un navegador: solo páginas públicas, con pausas de cortesía
(nada de APIs internas). El buscador solo deja recorrer los primeros 10.000 resultados de cada
consulta (página 476); desde la 477 repite siempre la misma página.
"""
from __future__ import annotations
import json
import re
import time

import requests

BASE = "https://www.fincaraiz.com.co"
ORDEN_RECIENTE = 3              # ordenListado: 0 mayor precio, 1 menor precio, 2 popularidad,
                                #               3 más recientes, 4 menor m², 5 mayor m²
POR_PAGINA = 21
MAX_RESULTADOS = 10_000         # tope del buscador por consulta
ULTIMA_PAGINA = MAX_RESULTADOS // POR_PAGINA          # 476
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
_NEXT = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def url_pagina(ruta: str, n: int, orden: int = ORDEN_RECIENTE) -> str:
    """ruta = 'venta', 'venta/cali/valle-del-cauca', 'venta/apartamentos/zona-norte/bogota'…"""
    return f"{BASE}/{ruta.strip('/')}" + (f"/pagina{n}" if n > 1 else "") + f"?&ordenListado={orden}"


def pedir(sesion: requests.Session, ruta: str, n: int, orden: int = ORDEN_RECIENTE,
          reintentos: int = 3) -> tuple[list[dict], dict, float]:
    """(avisos crudos del JSON, paginatorInfo, segundos) de la página n de `ruta`."""
    for intento in range(1, reintentos + 1):
        try:
            t = time.time()
            r = sesion.get(url_pagina(ruta, n, orden), headers=UA, timeout=30)
            r.raise_for_status()
            m = _NEXT.search(r.text)
            if not m:
                raise ValueError(f"Sin __NEXT_DATA__ en {r.url}")
            sf = json.loads(m.group(1))["props"]["pageProps"]["fetchResult"]["searchFast"]
            return sf.get("data") or [], sf.get("paginatorInfo") or {}, time.time() - t
        except (requests.RequestException, ValueError, KeyError):
            if intento == reintentos:
                raise
            time.sleep(5 * intento)                     # espera creciente antes de reintentar
    raise RuntimeError("inalcanzable")


def _nombres(loc: dict, nivel: str) -> str | None:
    vals = loc.get(nivel) or []
    return ", ".join(v.get("name", "") for v in vals) or None


def aplanar(a: dict) -> dict:
    """Un aviso del JSON → una fila plana con los campos útiles + el JSON original completo."""
    loc = a.get("locations") or {}
    precio = a.get("price") or {}
    admin = a.get("commonExpenses") or {}
    ficha = {f.get("field"): f.get("value") for f in (a.get("technicalSheet") or [])}
    tag = a.get("tag") or {}
    estado = ficha.get("construction_state_name")
    # Etiquetas compatibles con las del scraper actual (las usa la alerta)
    etiquetas = [e for e in (estado, "Proyecto" if a.get("isProject") else None,
                             tag.get("tag_label")) if e]
    return {
        "id_inmueble": str(a.get("id")),
        "titulo": a.get("title"),
        "url": BASE + (a.get("link") or ""),
        "imagen": a.get("img"),
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
        "estado": estado,
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
        "destacado": a.get("highlight"),
        "etiquetas": json.dumps(etiquetas, ensure_ascii=False),
        "amenidades": "|".join(f.get("name", "") for f in (a.get("facilities") or [])) or None,
        "n_imagenes": a.get("image_count"),
        "duplicados": len(a.get("duplicated") or []),
        "codigo_anunciante": a.get("code2"),
        "descripcion": a.get("description"),
        "json_crudo": json.dumps(a, ensure_ascii=False),     # el aviso tal cual, por si mañana hace falta otro campo
    }
