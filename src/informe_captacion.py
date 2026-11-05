"""
Informe de precio para captar (producto 1 de E7): un PDF de 2 páginas sobre UN inmueble, para que el agente
le muestre al propietario a qué precio salir al mercado.

  - Precio estimado por el modelo (src/modelo_precio.py) con su rango del 80% y su confianza.
  - Los comparables: viviendas parecidas en venta hoy (mismos criterios que la alerta, src/oportunidades.py:
    mismo tipo y estrato, área ±30%, ±1 habitación; en el barrio, a 1 km o en la zona), cuánto piden, cuántos
    días llevan publicadas y cuántas ya bajaron de precio.
  - Cuánto tardaron en retirarse las parecidas que ya salieron (bajas del barrido, src/consolidar.py).
  - Arriendo estimado y rentabilidad bruta al precio sugerido (src/rentabilidad.py).
  - Un rango de salida: donde se cruzan el rango del modelo y la mitad central de los comparables.

El inmueble puede ser un aviso ya publicado (FincaRaíz o Metrocuadrado: por su enlace o su id; se compara
además su precio pedido) o uno que todavía no está publicado (su ficha a mano). Los comparables son de los dos
portales, sin repetidos: de Metrocuadrado solo los que no están también en FincaRaíz (registro único,
src/unificar.py), y nunca el propio inmueble en el otro portal. Metrocuadrado no da la fecha de publicación ni
tenemos sus bajadas de precio: los días publicados y "bajó de precio" son de los de FincaRaíz.
Es una estimación del precio de mercado pedido, no un avalúo (Ley 1673 de 2013).

Uso:
    python -m src.informe_captacion --aviso https://www.fincaraiz.com.co/.../193004385
    python -m src.informe_captacion --ciudad Cali --tipo Apartamento --m2 85 --habitaciones 3 --banos 2 \\
        --estrato 5 --barrio "El Ingenio" [--parqueaderos 1] [--piso 4] [--antiguedad 3] [--administracion 350000]
    … --barrio "3.3712, -76.5321"            # o un enlace de Google Maps (también maps.app.goo.gl/…)
    … [--enviar admin]                       # por Telegram al administrador (o a los agentes)
"""
from __future__ import annotations
import argparse
import html
import json
import re

import numpy as np
import pandas as pd

from src import oportunidades as op, rentabilidad, zonas
from src.config import BASE_DIR
from src.consolidar import rutas
from src.informe_semanal import _fecha, _mill, _n, _pct, _tile

OUT_DIR = BASE_DIR / "reports" / "captacion"
MAX_COMPARABLES = 8         # los más parecidos, en la tabla
MIN_RETIRADOS = 5           # menos que esto no dice nada de cuánto tardan en salir
INICIO_BAJAS = "2026-10-28"  # primeras bajas del barrido (las anteriores son del rescate, sin fecha real)
ANTIGUEDAD = {"menos de 1 año": 1, "1 a 8 años": 2, "9 a 15 años": 3, "16 a 30 años": 4, "más de 30 años": 5}
AZUL, AZUL_CLARO, GRIS, TINTA, TINTA_2 = "#1c5cab", "#cde2fb", "#898781", "#0b0b0b", "#52514e"


# ── El inmueble ────────────────────────────────────────────────────────────────────────────────────────


def _hasta() -> pd.Timestamp:
    return pd.Timestamp(json.loads(rutas("venta").meta.read_text())["ultimo_barrido"])


def ficha_de_aviso(ref: str) -> pd.Series:
    """La ficha de un aviso publicado, por su enlace o su id (FincaRaíz o Metrocuadrado)."""
    ident = str(ref).rstrip("/").split("/")[-1].split("?")[0]
    for portal, raiz in (("FincaRaíz", "venta"), ("Metrocuadrado", "m2_venta")):
        a = pd.read_parquet(rutas(raiz).avisos)
        fila = a[a["id_inmueble"].astype(str) == ident]
        if len(fila):
            fila = zonas.corregir_ubicacion(fila, solo_imposibles=raiz == "venta")
            f = fila.iloc[0].copy()
            f["portal"] = portal
            f["ciudad"] = zonas.canonica(f["ciudad"])
            return f
    raise SystemExit(f"❌ No encontré el aviso {ident} en FincaRaíz ni en Metrocuadrado (¿se retiró?)")


class BarrioNoEncontrado(Exception):
    """El barrio escrito no existe en la ciudad, o puede ser varios: trae las sugerencias."""

    def __init__(self, barrio: str, ciudad: str, sugerencias: list[str], ambiguo: bool = False):
        self.barrio, self.ciudad, self.sugerencias, self.ambiguo = barrio, ciudad, sugerencias, ambiguo
        super().__init__(self.mensaje())

    def mensaje(self) -> str:
        lista = "\n".join(f"• {s}" for s in self.sugerencias)
        if self.ambiguo:
            return (f"❓ Informe de precio: «{self.barrio}» puede ser varios barrios de {self.ciudad}:\n{lista}\n"
                    f"Vuelve a lanzarlo con uno de estos nombres.")
        return (f"❓ Informe de precio: no encontré el barrio «{self.barrio}» en {self.ciudad}."
                + (f" ¿Quisiste decir…?\n{lista}\n" if lista else "\n")
                + "Vuelve a lanzarlo con el nombre exacto, sus coordenadas («3.3712, -76.5321») o el enlace de "
                  "Google Maps del punto.")


class UbicacionInvalida(BarrioNoEncontrado):
    """Coordenadas o enlace de mapa que no se pueden leer, o que caen lejos de la ciudad elegida."""

    def __init__(self, texto: str, ciudad: str, motivo: str):
        self.texto, self.motivo = texto, motivo
        super().__init__(texto, ciudad, [])

    def mensaje(self) -> str:
        return (f"❓ Informe de precio: {self.motivo} («{self.texto[:120]}»).\n"
                "Escribe el barrio, las coordenadas como «3.3712, -76.5321» o pega el enlace de Google Maps del punto.")


# Coordenadas escritas ("3.3712, -76.5321") o en un enlace de Google Maps: el pin (!3d…!4d…), ?q=/ll=/query= o @
COORDS = [re.compile(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)"),
          re.compile(r"[?&](?:q|ll|query|destination)=(-?\d+\.\d+)(?:,|%2C)\s*(-?\d+\.\d+)"),
          re.compile(r"@(-?\d+\.\d+),(-?\d+\.\d+)"),
          re.compile(r"^\s*\(?(-?\d{1,2}\.\d+)\s*[,;\s]\s*(-?\d{1,3}\.\d+)\)?\s*$")]


def coordenadas(texto: str | None, ciudad: str) -> tuple[float, float] | None:
    """(lat, lon) si `texto` son coordenadas o un enlace de Google Maps (también los cortos, maps.app.goo.gl, que
    se abren para ver a dónde llevan); None si es otra cosa (un nombre de barrio). UbicacionInvalida si son
    coordenadas pero no se pueden usar: fuera de Colombia o a más de LEJOS_CIUDAD km de la ciudad."""
    t = (texto or "").strip()
    if not t:
        return None
    es_enlace = re.match(r"https?://", t)
    if es_enlace and re.search(r"(goo\.gl|g\.co)/", t):
        import urllib.request
        try:
            pedido = urllib.request.Request(t, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(pedido, timeout=20) as r:
                t = r.geturl() + " " + r.read(200_000).decode("utf-8", "ignore")
        except OSError as e:
            raise UbicacionInvalida(texto, ciudad, f"no pude abrir el enlace ({e})") from None
    for patron in COORDS:
        if m := patron.search(t):
            lat, lon = float(m.group(1)), float(m.group(2))
            break
    else:
        if es_enlace:
            raise UbicacionInvalida(texto, ciudad, "no encontré coordenadas en el enlace")
        return None
    la0, la1, lo0, lo1 = zonas.COLOMBIA
    if not (la0 <= lat <= la1 and lo0 <= lon <= lo1):
        raise UbicacionInvalida(texto, ciudad, f"las coordenadas {lat:.5f}, {lon:.5f} no están en Colombia "
                                               "(¿latitud y longitud al revés?)")
    from shapely.geometry import Point
    toda, _, _ = zonas._ubicacion(ciudad)
    km = toda.distance(Point(lon, lat)) * 111
    if km > zonas.LEJOS_CIUDAD:
        raise UbicacionInvalida(texto, ciudad, f"el punto {lat:.5f}, {lon:.5f} queda a {km:.0f} km de {ciudad}")
    return lat, lon


SIMILITUD_SEGURA = 0.85     # un error de escritura ("El Ingenip"): se acepta e informa, si no hay otro casi igual
SIMILITUD_SUGERIR = 0.6     # menos parecido: solo como sugerencia


def resolver_barrio(barrio: str, ciudad: str) -> tuple[object, str, bool]:
    """(punto, nombre oficial, interpretado) del barrio escrito. En orden: el nombre (sin tildes, artículos ni
    paréntesis; cada parte de un nombre compuesto), uno que lo contenga o esté contenido en él, uno casi igual
    (errores de escritura) y, al final, una zona o un lugar de OSM con ese nombre. Si puede ser varios barrios
    distintos, o ninguno, lanza BarrioNoEncontrado con sugerencias (mejor preguntar que adivinar)."""
    import difflib
    from shapely.geometry import shape
    nombres_zona = zonas.nombres(ciudad)
    entradas, completos = [], []                    # (clave, nombre, zona, polígono)
    completo = lambda n: " ".join(t for t in re.sub(r"[^a-z0-9 ]", " ", zonas.normalizar(n)).split() if t not in zonas.VACIAS)
    for f in zonas.cargar("barrios", ciudad)["features"]:
        n, g = f["properties"]["nombre"], shape(f["geometry"])
        z = nombres_zona.get(f["properties"].get("comuna"), "")
        entradas += [(k, n, z, g) for k in zonas._alias(n)]
        completos.append((completo(n), n, z, g))    # el nombre entero, con lo que va entre paréntesis
    etiqueta = lambda e: f"{e[1]} ({e[2]})" if e[2] else e[1]
    # Una sugerencia copiada tal cual trae la zona entre paréntesis al final: "Santa Barbara (Santa Fe)"
    nombre, zona = barrio, None
    m = re.match(r"^(.*)\(([^()]*)\)\s*$", barrio)
    if m and any(zonas.normalizar(m.group(2)) == zonas.normalizar(z) for z in nombres_zona.values() if z):
        nombre, zona = m.group(1).strip(), zonas.normalizar(m.group(2))
        entradas = [e for e in entradas if zonas.normalizar(e[2]) == zona]
        completos = [e for e in completos if zonas.normalizar(e[2]) == zona]
    k = zonas.clave(nombre)

    def elegir(cand: list, interpretado: bool):
        distintos = {(n, z) for _, n, z, _ in cand}
        if len(distintos) == 1:
            _, n, _, g = max(cand, key=lambda e: e[3].area)
            return g.representative_point(), n, interpretado
        raise BarrioNoEncontrado(barrio, ciudad, sorted({etiqueta(e) for e in cand})[:8], ambiguo=True)

    exactos = [e for e in completos if e[0] == completo(nombre)]    # el nombre entero: "El Ingenio", "Pance (cabecera)"
    if exactos:
        return elegir(exactos, False)
    # Una parte del nombre ("Pance" de "Pance (cabecera)", "Cristales" de "Tejares - Cristales") o uno que lo
    # contenga: si son varios barrios, se pregunta ("Pance" también es "Parcelaciones Pance")
    contiene = [e for e in entradas if e[0] == k or zonas._coinciden(k, e[0])]
    if contiene:
        return elegir(contiene, True)
    claves = sorted({e[0] for e in entradas})
    parecidas = difflib.get_close_matches(k, claves, n=5, cutoff=SIMILITUD_SUGERIR)
    ratio = [difflib.SequenceMatcher(None, k, c).ratio() for c in parecidas]
    if parecidas and ratio[0] >= SIMILITUD_SEGURA and (len(ratio) == 1 or ratio[0] - ratio[1] >= 0.05):
        return elegir([e for e in entradas if e[0] == parecidas[0]], True)
    _, _, tablas = zonas._ubicacion(ciudad)         # zonas (corregimientos…) y lugares de OSM, con su nombre
    for t in tablas:
        if k in t:
            return t[k], barrio, True
    raise BarrioNoEncontrado(barrio, ciudad, [etiqueta(e) for c in parecidas for e in entradas if e[0] == c][:6])


def ficha_manual(ciudad: str, tipo: str, m2: float, habitaciones: float | None = None, banos: float | None = None,
                 estrato: float | None = None, barrio: str | None = None, lat: float | None = None,
                 lon: float | None = None, parqueaderos: float | None = None, piso: float | None = None,
                 antiguedad: int | None = None, administracion: float | None = None,
                 descripcion: str = "") -> pd.Series:
    """La ficha de un inmueble que no está publicado. Sin coordenadas, se ubica en el barrio que se nombra;
    `barrio` también puede traer las coordenadas o un enlace de Google Maps (el formulario tiene un solo campo)."""
    if (lat is None or lon is None) and (punto := coordenadas(barrio, ciudad)):
        (lat, lon), barrio = punto, None
    por_barrio = lat is None or lon is None
    if por_barrio:
        if not barrio:
            raise SystemExit("❌ Hace falta el barrio o las coordenadas (--lat y --lon)")
        punto, oficial, interpretado = resolver_barrio(barrio, ciudad)
        lat, lon = punto.y, punto.x
        escrito, barrio = (barrio if interpretado else None), oficial
    else:
        escrito = None
    venta = pd.read_parquet(rutas("venta").avisos, columns=["ciudad", "departamento"])
    deptos = venta.loc[venta["ciudad"].map(zonas.canonica) == ciudad, "departamento"].mode()
    return pd.Series({
        "id_inmueble": "", "portal": None, "url": None, "precio": np.nan, "tipo": tipo, "m2": m2,
        "m2_construido": np.nan, "m2_terreno": np.nan, "habitaciones": habitaciones, "banos": banos,
        "parqueaderos": parqueaderos, "piso": piso, "estrato": estrato, "antiguedad": antiguedad,
        "administracion": administracion, "latitud": lat, "longitud": lon, "ciudad": ciudad,
        "departamento": deptos.iloc[0] if len(deptos) else None, "barrio": barrio, "zona": None, "comuna": None,
        "estado": None, "amenidades": "", "descripcion": descripcion, "n_imagenes": np.nan, "etiquetas": "[]",
        "fecha_publicacion": None, "ubicacion": "barrio" if por_barrio else "aviso", "barrio_escrito": escrito})


def _como_aviso(f: pd.Series, ciudad: str) -> pd.Series:
    """La ficha con el formato de la alerta (src/oportunidades.py), con su barrio y zona oficiales."""
    num = lambda c: pd.to_numeric(f.get(c), errors="coerce")
    d = pd.DataFrame([{"Latitud": num("latitud"), "Longitud": num("longitud"), "Barrio": f.get("barrio")}])
    z = zonas.asignar_zonas(d, ciudad).iloc[0] if zonas.tiene_zonas(ciudad) else {"barrio_oficial": None, "comuna": None}
    return pd.Series({"id_inmueble": str(f.get("id_inmueble") or ""), "Tipo_propiedad": f["tipo"],
                      "Area_m2": num("m2"), "Habitaciones": num("habitaciones"), "Baños": num("banos"),
                      "Estrato": num("estrato") if 1 <= (num("estrato") or 0) <= 6 else np.nan,
                      "Latitud": num("latitud"), "Longitud": num("longitud"),
                      "barrio_oficial": z["barrio_oficial"], "comuna": z["comuna"], "Barrio": f.get("barrio")})


def estimar(f: pd.Series) -> dict:
    """Precio estimado por el modelo: estimado, rango p10–p90 (80%) y confianza."""
    from src.modelo_precio import ModeloPrecio, preparar
    r = ModeloPrecio.cargar().predecir(preparar(pd.DataFrame([f]))).iloc[0]
    return {"estimado": float(r["precio_estimado"]), "p10": float(r["precio_p10"]), "p90": float(r["precio_p90"]),
            "pm2": float(r["pm2_estimado"]), "confianza": r["confianza"]}


# ── El mercado alrededor ───────────────────────────────────────────────────────────────────────────────


def en_venta(ciudad: str, f: pd.Series) -> pd.DataFrame:
    """Viviendas en venta de los dos portales con el formato de la alerta (columna `portal`), sin repetidos:
    de Metrocuadrado solo las que no están también en FincaRaíz, y sin el propio inmueble `f` en ninguno."""
    from src import fuente_http, unificar, viz_map
    fr = op.cargar(ciudad).assign(portal="FincaRaíz")
    try:
        avisos = unificar.unificar(ciudad)[0]
    except FileNotFoundError:               # sin el estado de Metrocuadrado
        return fr
    con_fr = avisos.loc[avisos["portal"] == "fincaraiz", "id_unificado"]
    solo = avisos[(avisos["portal"] == "metrocuadrado") & ~avisos["id_unificado"].isin(con_fr)]
    solo = solo.drop_duplicates("id_inmueble").set_index("id_inmueble")
    m2 = fuente_http.dataset_activo("m2_venta")
    m2 = m2[m2["id_inmueble"].isin(solo.index)].copy()
    m2["Latitud"] = m2["id_inmueble"].map(solo["latitud"])        # la ubicación corregida (src/unificar.py)
    m2["Longitud"] = m2["id_inmueble"].map(solo["longitud"])
    m2["Ciudad"] = m2["Ciudad"].map(zonas.canonica)                # "Bogotá D.C." → "Bogotá"
    try:
        m2 = op.cargar(ciudad, viz_map.limpiar_ciudad(m2, ciudad)).assign(portal="Metrocuadrado")
    except ValueError:                      # ninguno válido en la ciudad
        m2 = fr.iloc[:0]
    todo = pd.concat([fr, m2], ignore_index=True)
    if f.get("portal"):                     # el propio inmueble, en los dos portales
        clave = ("fincaraiz:" if f["portal"] == "FincaRaíz" else "metrocuadrado:") + str(f["id_inmueble"])
        grupo = avisos.loc[avisos["clave"] == clave, "id_unificado"]
        mismo = avisos.loc[avisos["id_unificado"].isin(grupo), "id_inmueble"]
        todo = todo[~todo["id_inmueble"].isin(mismo)]
    return todo.reset_index(drop=True)


def datos(f: pd.Series) -> dict:
    """Todo lo que lleva el informe."""
    ciudad = zonas.canonica(f["ciudad"])
    hasta = _hasta()
    c = _como_aviso(f, ciudad)
    pool = en_venta(ciudad, f)
    comps, nivel = op._comparables(c, pool)
    if nivel is None:                   # pocos: se usan los del barrio que haya, con confianza baja
        comps, nivel = op._comparables(c, pool, minimo=3)
    comps = comps.assign(pm2=comps["Precio"] / comps["Area_m2"],
                         dias=(hasta - comps["first_seen"]).dt.days, parecido=op.parecido(c, comps))
    hist = op.historial(hasta)
    comps["bajo"] = comps["id_inmueble"].map(hist["n_bajadas"]).fillna(0) > 0 if len(hist) else False
    comps["bajo"] = comps["bajo"].where(comps["portal"] == "FincaRaíz")      # de Metrocuadrado no lo sabemos

    # Los parecidos que ya se retiraron: cuánto tardaron
    from src import fuente_http, viz_map
    ret = viz_map.limpiar_ciudad(fuente_http.dataset_activo("venta", retirados=True), ciudad)
    ret = ret[ret["Tipo_propiedad"].isin(op.TIPOS) & (ret["fecha_baja"] >= INICIO_BAJAS)]
    for col in ("Estrato", "Habitaciones"):
        ret[col] = pd.to_numeric(ret[col], errors="coerce")
    retirados, nivel_ret = op._comparables(c, ret, minimo=MIN_RETIRADOS)
    retirados = retirados.assign(dias=(retirados["fecha_baja"] - retirados["first_seen"]).dt.days).dropna(subset=["dias"])

    m = estimar(f)
    m2 = c["Area_m2"]
    q = comps["pm2"].quantile([.25, .5, .75]) if len(comps) else pd.Series(np.nan, index=[.25, .5, .75])
    bajo, alto = max(m["p10"], q[.25] * m2), min(m["p90"], q[.75] * m2)
    rango = (bajo, alto) if bajo < alto else (m["p10"], m["p90"])   # si no se cruzan, el del modelo

    # Rentabilidad al precio estimado
    rent = None
    if rentabilidad.disponible():
        venta = pd.DataFrame([c.to_dict() | {"Precio": m["estimado"]}])
        r = rentabilidad.estimar(venta, rentabilidad.arriendos(ciudad)).iloc[0]
        rent = r if pd.notna(r.get("nivel_arr")) else None

    pedido = pd.to_numeric(f.get("precio"), errors="coerce")
    return {"f": f, "c": c, "ciudad": ciudad, "hasta": hasta, "modelo": m, "rango": rango, "nivel": nivel,
            "comps": comps.sort_values("parecido"), "q": q, "retirados": retirados, "nivel_ret": nivel_ret,
            "rent": rent, "pedido": pedido if pd.notna(pedido) else None,
            "dias_pedido": (hasta - pd.to_datetime(f.get("fecha_publicacion"), errors="coerce")).days
            if pd.notna(pd.to_datetime(f.get("fecha_publicacion"), errors="coerce")) else None,
            "zona": op._en_zona(pd.Series({"zona_nombre": zonas.nombres(ciudad).get(c["comuna"]) if zonas.tiene_zonas(ciudad) else None,
                                           "comuna": c["comuna"]})) if pd.notna(c["comuna"]) else None}


# ── El gráfico: dónde queda el precio frente a los parecidos ──────────────────────────────────────────


def grafico(d: dict, w: float = 500, h: float = 118) -> str:
    """Una fila de puntos: el precio por m² de cada parecido; la franja es el rango del modelo; la línea, el
    estimado; el rombo, el precio pedido (si el inmueble ya está publicado). SVG con atributos en línea
    (WeasyPrint no aplica el CSS de la página dentro del SVG)."""
    comps, m, m2 = d["comps"], d["modelo"], d["c"]["Area_m2"]
    valores = [*comps["pm2"], m["p10"] / m2, m["p90"] / m2] + ([d["pedido"] / m2] if d["pedido"] else [])
    lo, hi = np.nanpercentile(valores, 1) * 0.92, np.nanpercentile(valores, 99) * 1.06
    izq, der, base = 14, w - 14, 70
    x = lambda v: izq + (min(max(v, lo), hi) - lo) / (hi - lo) * (der - izq)
    t = []
    # Franja del modelo (80%) y su etiqueta
    t.append(f'<rect x="{x(m["p10"] / m2):.1f}" y="{base - 30}" width="{x(m["p90"] / m2) - x(m["p10"] / m2):.1f}" '
             f'height="42" rx="3" fill="{AZUL_CLARO}"/>')
    # Puntos de los parecidos (con anillo del color de fondo para que no se fundan)
    rng = np.random.default_rng(0)
    for v, j in zip(comps["pm2"], rng.uniform(-9, 9, len(comps))):
        t.append(f'<circle cx="{x(v):.1f}" cy="{base - 9 + j:.1f}" r="3.4" fill="{GRIS}" stroke="#ffffff" stroke-width="1"/>')
    # Estimado
    xe = x(m["pm2"])
    t.append(f'<line x1="{xe:.1f}" x2="{xe:.1f}" y1="{base - 34}" y2="{base + 14}" stroke="{AZUL}" stroke-width="2"/>')
    t.append(f'<text x="{xe:.1f}" y="{base - 38}" text-anchor="middle" font-family="Inter, sans-serif" font-size="7.4" '
             f'font-weight="600" fill="{TINTA}">Estimado {_mill(m["pm2"], 1)}/m²</text>')
    # Precio pedido
    if d["pedido"]:
        xp = x(d["pedido"] / m2)
        t.append(f'<path d="M{xp:.1f},{base + 14} l5,6 l-5,6 l-5,-6 z" fill="{TINTA}"/>')
        t.append(f'<text x="{xp:.1f}" y="{base + 36}" text-anchor="middle" font-family="Inter, sans-serif" font-size="7.2" '
                 f'fill="{TINTA}">Pedido {_mill(d["pedido"] / m2, 1)}/m²</text>')
    # Eje
    paso = 0.5e6 if hi - lo < 4e6 else 1e6
    v = np.ceil(lo / paso) * paso
    t.append(f'<line x1="{izq}" x2="{der}" y1="{base + 14}" y2="{base + 14}" stroke="#e1e0d9" stroke-width="1"/>')
    while v <= hi:
        t.append(f'<text x="{x(v):.1f}" y="{base + 24}" text-anchor="middle" font-family="Inter, sans-serif" '
                 f'font-size="6.6" fill="{GRIS}">{f"{v / 1e6:.1f}".replace(".", ",")}</text>')
        v += paso
    return f'<svg width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{"".join(t)}</svg>'


# ── El HTML ─────────────────────────────────────────────────────────────────────────────────────────────


def _ficha_txt(f: pd.Series) -> str:
    n = lambda c: pd.to_numeric(f.get(c), errors="coerce")
    partes = [f"{n('m2'):.0f} m²"]
    for c, s, p in (("habitaciones", "habitación", "habitaciones"), ("banos", "baño", "baños"),
                    ("parqueaderos", "parqueadero", "parqueaderos")):
        if pd.notna(n(c)) and n(c) > 0:
            partes.append(f"{n(c):.0f} {s if n(c) == 1 else p}")
    if pd.notna(n("estrato")):
        partes.append(f"estrato {n('estrato'):.0f}")
    if pd.notna(n("piso")) and n("piso") > 0:
        partes.append(f"piso {n('piso'):.0f}")
    return " · ".join(partes)


def _redondear(x: float) -> float:
    """Precios de salida redondeados como se publican: a 5 M (a 1 M si es menos de 200 M)."""
    paso = 1e6 if x < 2e8 else 5e6
    return round(x / paso) * paso


def informe_html(d: dict) -> str:
    f, c, m, comps, q, ret = d["f"], d["c"], d["modelo"], d["comps"], d["q"], d["retirados"]
    m2 = c["Area_m2"]
    lugar = c["barrio_oficial"] if pd.notna(c["barrio_oficial"]) else (f.get("barrio") or d["ciudad"])
    titulo = f"{f['tipo']} de {pd.to_numeric(f['m2']):.0f} m² en {lugar}"
    bajo, alto = (_redondear(x) for x in d["rango"])
    donde = {"barrio": "en el barrio", "1 km": "a menos de 1 km", "comuna": d["zona"] or "en la zona"}.get(d["nivel"], "cerca")
    n_bajo = int((comps["bajo"] == True).sum()) if len(comps) else 0   # noqa: E712 (vacío = no se sabe)
    n_fr = int((comps["portal"] == "FincaRaíz").sum()) if len(comps) else 0
    n_m2 = len(comps) - n_fr

    tiles = [_tile("Precio estimado", _mill(_redondear(m["estimado"]), 0), f"confianza {m['confianza'].lower()}"),
             _tile("Rango de salida sugerido", f"{_mill(bajo, 0)} – {_mill(alto, 0)}".replace(" M –", " –"),
                   "donde coinciden el modelo y los similares"),
             _tile("Similares en venta", _n(len(comps)), f"{donde}; piden {_mill(q[.5], 1)}/m² (mediana)"
                   if len(comps) else "sin similares cerca")]
    if d["rent"] is not None:
        r = d["rent"]
        tiles.append(_tile("Rentabilidad bruta", _pct(r["rentab_bruta"], 1),
                           f"arriendo estimado {_mill(r['canon_est'], 1)}/mes"))
    else:
        tiles.append(_tile("Días publicados", f"{comps['dias'].median():.0f}" if comps["dias"].notna().any() else "—",
                           "mediana de los similares en venta"))

    puntos = [f"Con su ficha, el modelo estima <b>{_mill(_redondear(m['estimado']), 0)}</b> "
              f"({_mill(m['pm2'], 1)}/m²); 8 de cada 10 inmuebles así se piden entre "
              f"{_mill(_redondear(m['p10']), 0)} y {_mill(_redondear(m['p90']), 0)}."]
    if len(comps):
        portales = f" ({n_fr} de FincaRaíz y {n_m2} solo en Metrocuadrado)" if n_m2 else ""
        de_fr = f"; los de FincaRaíz llevan" if n_m2 else "; llevan"
        bajaron = (f"{n_bajo} de ellos ya bajaron de precio." if n_bajo else "ninguno ha bajado de precio.")
        dias = (f"{de_fr} publicados {comps['dias'].median():.0f} días (mediana) y {bajaron}"
                if comps["dias"].notna().any() else ".")
        puntos.append(f"Los <b>{len(comps)} similares</b> {donde}{portales} piden entre {_mill(q[.25] * m2, 0)} y "
                      f"{_mill(q[.75] * m2, 0)} la mitad central (a su área){dias}")
    if len(ret) >= MIN_RETIRADOS:
        puntos.append(f"De los parecidos que ya se retiraron del portal desde el {_fecha(pd.Timestamp(INICIO_BAJAS), año=False)}, "
                      f"la mitad lo hizo antes de <b>{ret['dias'].median():.0f} días</b> de publicado.")
    if d["pedido"]:
        dif = d["pedido"] / m["estimado"] - 1
        puntos.append(f"Hoy se pide <b>{_mill(d['pedido'], 0)}</b>: {_pct(abs(dif))} "
                      f"{'sobre' if dif > 0 else 'bajo'} el estimado"
                      + (f", y lleva {d['dias_pedido']} días publicado." if d["dias_pedido"] is not None else "."))

    filas = "".join(
        f"<tr><td>{html.escape(str(r['barrio_oficial'] if pd.notna(r['barrio_oficial']) else r['Barrio']))}</td>"
        f"<td>{r['Area_m2']:.0f} m²" + (f" · {r['Habitaciones']:.0f} hab" if pd.notna(r['Habitaciones']) else "")
        + (f" · {r['Baños']:.0f} baños" if pd.notna(r.get('Baños')) else "") + "</td>"
        f"<td class='num'>{_mill(r['Precio'], 0)}</td><td class='num'>{_mill(r['pm2'], 1)}</td>"
        f"<td class='num'>{'' if pd.isna(r['dias']) else f'{r.dias:.0f}'}</td>"
        f"<td class='num'>{'' if pd.isna(r['bajo']) else 'sí' if r['bajo'] else '—'}</td>"
        f"<td><a href='{html.escape(str(r['URL detalle']))}'>{r['portal']} ↗</a></td></tr>"
        for _, r in comps.head(MAX_COMPARABLES).iterrows())
    tabla = (f"""<table><thead><tr><th>Barrio</th><th>Ficha</th><th class="num">Precio</th><th class="num">Por m²</th>
      <th class="num">Días publicado</th><th class="num">Bajó de precio</th><th></th></tr></thead><tbody>{filas}</tbody></table>"""
             if len(comps) else '<p class="vacio">No hay similares en venta cerca: el estimado se apoya solo en el modelo.</p>')

    if len(ret) >= MIN_RETIRADOS:
        r_txt = (f"<p>{len(ret)} viviendas parecidas se retiraron del portal desde el {_fecha(pd.Timestamp(INICIO_BAJAS))}: "
                 f"llevaban publicadas <b>{ret['dias'].median():.0f} días</b> (mediana; la cuarta parte más rápida, "
                 f"{ret['dias'].quantile(.25):.0f} días o menos). Retirarse no siempre es venderse: puede ser que el dueño "
                 f"desistió o cambió de inmobiliaria.</p>")
    else:
        r_txt = ("<p>Todavía hay pocas bajas de parecidos (el barrido las registra desde el 28 de octubre): esta sección "
                 "se llenará en las próximas semanas. Mientras tanto, los días que llevan publicados los similares dicen cuánto "
                 "tarda lo que no se vende.</p>")
    if d["rent"] is not None:
        r = d["rent"]
        rent_txt = (f"<p>Al precio estimado, el arriendo de un inmueble así rondaría <b>{_mill(r['canon_est'], 1)} al mes</b>: "
                    f"rentabilidad bruta de {_pct(r['rentab_bruta'], 1)} (rango {_pct(r['rentab_p25'], 1)}–{_pct(r['rentab_p75'], 1)}), "
                    f"con {r['n_arr']:.0f} arriendos similares; confianza {r['confianza_arr'].lower()}.</p>")
    else:
        rent_txt = "<p>No hay suficientes arriendos parecidos cerca para estimar el canon.</p>"

    ubic = ("Ubicado en el centro del barrio que se indicó: con la dirección exacta, el estimado puede cambiar."
            if f.get("ubicacion") == "barrio" else "")
    enlace = f' · <a href="{html.escape(str(f["url"]))}">ver el aviso ↗</a>' if f.get("url") else ""
    if f.get("barrio_escrito"):
        enlace += f' · barrio «{html.escape(str(f["barrio"]))}» (se escribió «{html.escape(str(f["barrio_escrito"]))}»)'
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Precio para captar · {html.escape(titulo)}</title>
<style>
  @page {{ size: A4; margin: 13mm 15mm 14mm;
    @bottom-left {{ content: "Informe de precio · {html.escape(d['ciudad'])} · {_fecha(d['hasta'])}"; font: 7pt Inter, sans-serif; color: #898781; }}
    @bottom-right {{ content: "Página " counter(page) " de " counter(pages); font: 7pt Inter, sans-serif; color: #898781; }} }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: Inter, "DejaVu Sans", sans-serif; font-size: 8.8pt; line-height: 1.45; color: #0b0b0b; margin: 0; }}
  .kicker {{ font-size: 7.2pt; letter-spacing: .08em; text-transform: uppercase; color: #898781; margin: 0 0 2pt; }}
  h1 {{ font-size: 20pt; font-weight: 600; margin: 0; letter-spacing: -.01em; }}
  .ficha {{ color: #52514e; margin: 2pt 0 10pt; }}
  h2 {{ font-size: 11.5pt; font-weight: 600; margin: 14pt 0 3pt; }}
  .sub {{ color: #52514e; margin: 0 0 6pt; }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6pt; margin-bottom: 8pt; }}
  .tile {{ background: #f9f9f7; border-radius: 6pt; padding: 6pt 9pt 7pt; }}
  .t-label {{ color: #52514e; font-size: 7.6pt; }}
  .t-value {{ font-size: 15pt; font-weight: 600; margin: 0; }}
  .t-note {{ color: #898781; font-size: 7pt; line-height: 1.3; }}
  .resumen {{ border-left: 2.5pt solid #2a78d6; padding: 1pt 0 1pt 9pt; margin: 6pt 0 0; }}
  .resumen ul {{ margin: 0; padding-left: 10pt; }} .resumen li {{ margin-bottom: 3pt; }}
  .ley {{ color: #52514e; font-size: 7.4pt; margin: 2pt 0 0; }}
  .ley i {{ display: inline-block; width: 9pt; height: 9pt; border-radius: 2pt; vertical-align: -1.5pt; margin: 0 3pt 0 9pt; }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th {{ text-align: left; font-weight: 500; color: #52514e; font-size: 7.4pt; padding: 0 4pt 4pt; border-bottom: 1px solid #e1e0d9; vertical-align: bottom; }}
  th.num, td.num {{ text-align: right; }}
  td {{ padding: 3.2pt 4pt; border-bottom: 1px solid #e1e0d9; }}
  td.num, td:last-child {{ white-space: nowrap; }}
  td:last-child {{ padding-left: 12pt; }}
  h2 {{ break-after: avoid; }} tr {{ break-inside: avoid; }}
  a {{ color: #2a78d6; font-weight: 600; text-decoration: none; }}
  .nota {{ color: #898781; font-size: 7pt; line-height: 1.4; margin-top: 10pt; }}
  .vacio {{ color: #52514e; }}
</style></head><body>
<p class="kicker">Informe de precio para captar · {html.escape(d['ciudad'])} · {_fecha(d['hasta'])}</p>
<h1>{html.escape(titulo)}</h1>
<p class="ficha">{html.escape(_ficha_txt(f))}{enlace}</p>
<div class="tiles">{''.join(tiles)}</div>
<div class="resumen"><ul>{''.join(f'<li>{p}</li>' for p in puntos)}</ul></div>

<h2>Dónde queda frente a los similares</h2>
<p class="sub">Precio pedido por m² de cada vivienda similar en venta hoy (puntos), el rango del modelo (franja) y su estimado (línea).</p>
{grafico(d)}
<p class="ley"><i style="background:{GRIS};border-radius:50%"></i>similar en venta<i style="background:{AZUL_CLARO}"></i>rango del modelo (80%)<i style="background:{AZUL};width:2pt"></i>estimado{'<i style="background:#0b0b0b;transform:rotate(45deg);width:6pt;height:6pt"></i>precio pedido hoy' if d['pedido'] else ''}</p>

<h2>Los similares en venta</h2>
{f"""<p class="sub">Los {min(len(comps), MAX_COMPARABLES)} más parecidos de {len(comps)} (mismo tipo{' y estrato' if pd.notna(c['Estrato']) else ''}, área ±30%, ±1 habitación, {donde}).{' Metrocuadrado no publica la fecha de publicación ni sabemos si bajaron de precio: en los suyos esas celdas van vacías.' if n_m2 else ''}</p>""" if len(comps) else ""}
{tabla}

<h2>Cuánto tardan en salir</h2>
{r_txt}

<h2>Si se compra para arrendar</h2>
{rent_txt}

<p class="nota">Estimación del precio de mercado <b>pedido</b> con avisos públicos de FincaRaíz y Metrocuadrado
(modelo y similares, sin repetir los que están en los dos), al {_fecha(d['hasta'])}. No es un avalúo (Ley 1673 de 2013): no reemplaza el de un avaluador inscrito
en el RAA. Los precios de cierre suelen ser algo menores que los pedidos. El modelo tiene un error mediano de ~10% en
apartamentos y ~15% en casas; el rango del 80% lo refleja. {ubic}</p>
</body></html>"""


def generar(f: pd.Series) -> tuple[str, dict]:
    """Escribe el .html y el .pdf; devuelve la ruta del PDF y los datos."""
    from weasyprint import HTML
    d = datos(f)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nombre = re.sub(r"[^a-z0-9]+", "_", zonas.normalizar(f"{d['ciudad']} {f['tipo']} {pd.to_numeric(f['m2']):.0f}m2 "
                                                            f"{d['c']['barrio_oficial'] or f.get('barrio') or ''}")).strip("_")
    base = OUT_DIR / f"{nombre}_{d['hasta']:%Y-%m-%d}"
    texto = informe_html(d)
    base.with_suffix(".html").write_text(texto, encoding="utf-8")
    HTML(string=texto, base_url=str(BASE_DIR)).write_pdf(base.with_suffix(".pdf"))
    print(f"✅ {base.with_suffix('.pdf')}")
    return str(base.with_suffix(".pdf")), d


def enviar(pdf: str, d: dict, a_quien: str) -> None:
    """Por Telegram: 'admin' (TELEGRAM_ADMIN_CHAT_ID) o 'agentes' (TELEGRAM_CHAT_ID, separados por coma)."""
    import os
    from src.notificar import documento
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chats = os.environ.get("TELEGRAM_ADMIN_CHAT_ID" if a_quien == "admin" else "TELEGRAM_CHAT_ID", "")
    if not (token and chats):
        raise SystemExit("❌ Falta TELEGRAM_BOT_TOKEN o el chat de destino")
    pie = f"Informe de precio · {d['f']['tipo']} de {pd.to_numeric(d['f']['m2']):.0f} m² · {d['ciudad']}"
    for chat in filter(None, map(str.strip, chats.split(","))):
        documento(pdf, token, chat, pie)
    print(f"✅ Enviado ({a_quien})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Informe de precio para captar un inmueble (PDF)")
    p.add_argument("--aviso", help="enlace o id de un aviso publicado (FincaRaíz o Metrocuadrado)")
    p.add_argument("--ciudad", default="Cali")
    p.add_argument("--tipo", choices=["Apartamento", "Apartaestudio", "Casa"])
    p.add_argument("--m2", type=float)
    p.add_argument("--habitaciones", type=float)
    p.add_argument("--banos", type=float)
    p.add_argument("--estrato", type=float)
    p.add_argument("--barrio", help="nombre del barrio, o sus coordenadas («3.3712, -76.5321»), o un enlace de Google Maps")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--parqueaderos", type=float)
    p.add_argument("--piso", type=float)
    p.add_argument("--antiguedad", choices=list(ANTIGUEDAD), help="rango de edad del inmueble")
    p.add_argument("--administracion", type=float, help="cuota mensual en pesos")
    p.add_argument("--descripcion", default="", help="texto como el de un aviso (opcional)")
    p.add_argument("--enviar", choices=["admin", "agentes"], help="lo envía por Telegram")
    a = p.parse_args()
    if a.aviso:
        ficha = ficha_de_aviso(a.aviso)
    elif a.tipo and a.m2:
        try:
            ficha = ficha_manual(a.ciudad, a.tipo, a.m2, a.habitaciones, a.banos, a.estrato, a.barrio, a.lat, a.lon,
                                 a.parqueaderos, a.piso, ANTIGUEDAD.get(a.antiguedad), a.administracion, a.descripcion)
        except BarrioNoEncontrado as e:            # mejor preguntar que adivinar: las sugerencias, por Telegram
            print(e.mensaje())
            if a.enviar:
                import os
                from src.notificar import telegram
                token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_ADMIN_CHAT_ID")
                if token and chat:
                    telegram(e.mensaje(), token, chat)
                    raise SystemExit(0)            # respondido: no hace falta el aviso genérico de fallo
            raise SystemExit(1)
    else:
        raise SystemExit("❌ Usa --aviso, o --tipo y --m2 (con --barrio o --lat/--lon) para un inmueble sin publicar")
    pdf, d = generar(ficha)
    if a.enviar:
        enviar(pdf, d, a.enviar)
