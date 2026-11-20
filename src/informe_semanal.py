"""
Informe semanal para los agentes: el pulso del mercado de vivienda de una ciudad, sus oportunidades y a quién
captar, en un PDF de cuatro páginas fácil de leer y de reenviar. Solo datos reales: lo que necesita semanas de
historia (tendencias, días hasta venderse, acierto de la alerta) se irá sumando.

  Página 1 · Pulso del mercado
    - viviendas en venta sin repetidos entre FincaRaíz y Metrocuadrado (src/unificar.py); publicadas,
      retiradas y que bajaron de precio en la semana, por inmueble en los dos portales (src/movimientos.py);
      arriendos de vivienda, canon y rentabilidad bruta típica. En el informe no se nombra ningún portal: es un
      solo mercado
    - el resumen en tres frases y el mapa de la ciudad por comuna o localidad, coloreado por precio por m²
  Página 2 · Por comuna o localidad (src/zonas.py, src/geo_ciudades.py; sin polígonos, la ciudad va sin ellas): viviendas en venta, precio mediano por m², días publicados (edad mediana de los
    avisos activos), canon mediano por m² y rentabilidad bruta típica (canon por m² × 12 ÷ precio por m²)
  Página 3 · Oportunidades: las de la alerta (src/oportunidades.py, mismos criterios), cada una con 3 de sus
    comparables, si el mismo inmueble está también en Metrocuadrado y a qué precio, el arriendo estimado y el
    entorno (OpenStreetMap, src/osm_pois.py: distancia al MIO, a un parque y a un centro comercial; sin conteos
    de comercios o colegios, que salen en 0 donde el mapa está incompleto). Cuatro tarjetas por página.
  Página 4 · Bajaron de precio y Para captar: viviendas con 4 a 12 meses publicadas que piden entre un 10% y un
    60% más que sus comparables (más que eso suele ser un error del aviso), las que más tiempo llevan, y quién las
    publica (quien_publica: «Particular», «1 inmobiliaria» o «N anunciantes», sin nombres)

Es una estimación del precio de mercado, no un avalúo (Ley 1673 de 2013). Sin fotos ni nombres de los
anunciantes: solo cifras y enlaces a los avisos públicos.

Uso:
    python -m src.informe_semanal [--ciudad Cali] [--fecha AAAA-MM-DD] [--enviar admin|agentes]
    → reports/informe/<ciudad>_<fecha>.pdf (y el .html del que sale); con --enviar, por Telegram al
      administrador (TELEGRAM_ADMIN_CHAT_ID, para revisarlo) o a los agentes (TELEGRAM_CHAT_ID)
"""
from __future__ import annotations
import argparse
import html
import json
import re

import numpy as np
import pandas as pd

from src.consolidar import AUSENCIAS_PARA_BAJA
from src import credito, movimientos, oportunidades as op, rentabilidad, tema, unificar, viz_map, zonas
from src.config import BASE_DIR

OUT_DIR = BASE_DIR / "reports" / "informe"
UMBRAL = 0.20               # descuento mínimo frente a los comparables (el de la alerta)
TOP = 4                     # oportunidades nuevas en la página 3 (tarjetas completas)
TOP_BAJADAS = 6
MIN_VIVIENDAS = 50          # una comuna entra en la tabla con al menos estas viviendas en venta
MAX_FILAS = 20              # y, si son más, las que tienen más oferta (la tabla cabe en una página)
MIN_ARRIENDOS = 15          # y su canon, con al menos estos arriendos
MIN_BARRIO = 5              # en el mapa, un barrio lleva color con al menos estas viviendas en venta
BARRIOS_POR_COMUNA = 2
CAPTAR_DIAS = (121, 365)    # para captar: entre 4 y 12 meses publicadas (más viejas suelen estar abandonadas)
CAPTAR_SOBRE = (0.10, 0.60)  # y entre un 10% y un 60% sobre sus comparables (más suele ser un error del aviso)
CAPTAR_FILAS = 10
NO_BARRIO = re.compile(r"^(área de expansión|area de expansion|suburbano)", re.I)   # en la tabla, solo barrios
PREFIJO = re.compile(r"^(urbanizaci[oó]n|parcelaci[oó]n(es)?|ciudadela|conjunto residencial)\s+", re.I)
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]    # secuencial azul, de menos a más
ALREDEDORES = "#d9d8d2"                                             # fuera del núcleo de un área: neutro


# ── Datos ─────────────────────────────────────────────────────────────────────

def _por_comuna(inm: pd.DataFrame, arr: pd.DataFrame, venta: pd.DataFrame) -> pd.DataFrame:
    """Viviendas en venta, precio mediano por m², días publicados, canon mediano por m² y rentabilidad bruta."""
    g = inm.dropna(subset=["comuna"]).groupby("comuna")
    t = pd.DataFrame({"viviendas": g.size(), "precio_m2": g["precio_m2"].median()})
    t["dias_pub"] = venta.groupby("comuna")["dias_pub"].median()
    ga = arr.dropna(subset=["comuna"]).groupby("comuna")
    t["arriendos"] = ga.size()
    t["canon_m2"] = ga["precio_m2"].median().where(t["arriendos"] >= MIN_ARRIENDOS)
    t["rentab"] = 12 * t["canon_m2"] / t["precio_m2"]
    urbanos = inm.dropna(subset=["comuna", "barrio_oficial"])
    urbanos = urbanos[~urbanos["barrio_oficial"].str.contains(NO_BARRIO, na=False)]
    barrios = urbanos.groupby("comuna")["barrio_oficial"].agg(
        lambda s: ", ".join(PREFIJO.sub("", b)[:28] for b in s.value_counts().index[:BARRIOS_POR_COMUNA]))
    # Sin barrios oficiales (los municipios vecinos de un área): los que más escriben los avisos
    del_aviso = inm.dropna(subset=["comuna", "Barrio"]).assign(
        Barrio=lambda x: x["Barrio"].astype(str).str.split(",").str[0].str.strip().str.title())
    t["barrios"] = barrios.combine_first(del_aviso.groupby("comuna")["Barrio"].agg(
        lambda s: ", ".join(b[:28] for b in s.value_counts().index[:BARRIOS_POR_COMUNA]))).fillna("")
    return t[t["viviendas"] >= MIN_VIVIENDAS].sort_values("precio_m2", ascending=False)   # la tabla: MAX_FILAS


EMPRESA = re.compile(r"inmobiliari|inmuebles|bienes|ra[ií]z|\bs\.?\s?a\.?\s?s\b|\bltda|grupo|asesor|broker|propiedad|"
                     r"constructor|realty|home|capital|gesti[oó]n|negocios|servicios|inversiones|consultor|proyectos", re.I)


GENERICAS_ANUNCIANTE = {"inmobiliaria", "inmobiliario", "inmobiliarios", "inmuebles", "sas", "s", "a", "sa", "ltda", "y",
                        "de", "del", "la", "el", "los", "las", "grupo", "group", "union", "propiedades", "bienes", "raiz",
                        "finca", "asesores", "asesorias", "broker", "brokers", "realty", "consultores", "gestion", "nc",
                        "bogota", "cali", "medellin", "barranquilla", "colombia", "cia", "co"}


def _clave_anunciante(nombre: str) -> str:
    """La primera palabra distintiva del nombre, para reconocer a la misma inmobiliaria escrita distinto en cada
    portal ("Unisa Inmobiliaria" y "UNISA UNION INMOBILIARIA" → "unisa"; "VELOZA INMOBILIARIA SAS" → "veloza")."""
    palabras = re.sub(r"[^a-z0-9 ]", " ", zonas.normalizar(nombre or "")).split()
    utiles = [w for w in palabras if w not in GENERICAS_ANUNCIANTE and len(w) >= 3]
    return utiles[0] if utiles else " ".join(palabras)


def quien_publica(sel: pd.DataFrame, avisos: pd.DataFrame) -> pd.Series:
    """Para cada aviso de FincaRaíz: 'Particular', '1 inmobiliaria' o 'N anunciantes' (contando los dos portales:
    el mismo inmueble en el registro único). Particular: Metrocuadrado lo marca «Persona natural», o en FincaRaíz el
    anunciante tiene 1 o 2 avisos en todo el país y su nombre no es de empresa (FincaRaíz no lo marca: es una pista)."""
    from src.consolidar import rutas
    nombres, particular = {}, set()
    for portal, raiz in unificar.PORTALES.items():
        cols = ["id_inmueble", "anunciante", "anunciante_id", "estado_aviso"] + (["tipo_anunciante"] if portal == "metrocuadrado" else [])
        a = pd.read_parquet(rutas(raiz).avisos, columns=cols)
        a = a[a["estado_aviso"] == "activo"]
        clave = portal + ":" + a["id_inmueble"].astype(str)
        nombre = a["anunciante"].fillna("").map(_clave_anunciante)
        nombres.update(zip(clave, nombre.where(nombre != "", a["anunciante_id"].astype(str))))
        if portal == "metrocuadrado":
            particular.update(clave[a["tipo_anunciante"].eq("Persona natural")])
        else:
            n = a.groupby("anunciante_id")["id_inmueble"].transform("size")
            particular.update(clave[(n <= 2) & ~a["anunciante"].fillna("").str.contains(EMPRESA)])
    grupo = avisos.set_index("clave")["id_unificado"]
    claves_de = avisos.groupby("id_unificado")["clave"].agg(list)

    def texto(id_fr: str) -> str:
        clave = f"fincaraiz:{id_fr}"
        claves = claves_de.get(grupo.get(clave), [clave])
        quienes = {nombres.get(k, k) for k in claves}
        hay_particular = any(k in particular for k in claves)
        if len(quienes) == 1:
            return "Particular" if hay_particular else "1 inmobiliaria"
        return f"{len(quienes)} anunciantes" + (" (uno particular)" if hay_particular else "")
    return sel["id_inmueble"].astype(str).map(texto)


def _entorno(sel: pd.DataFrame) -> pd.DataFrame:
    """Distancias y conteos de OpenStreetMap (src/osm_pois.py); vacío si no hay el parquet de servicios."""
    from src import osm_pois
    if sel.empty or not osm_pois.POIS_PARQUET.exists():
        return pd.DataFrame(index=sel.index)
    return osm_pois.features_servicios(sel)


def datos(ciudad: str = "Cali", fecha: str | None = None, dias: int = 7) -> dict:
    """Todo lo que lleva el informe de la semana que termina en `fecha` (por defecto, el último dato)."""
    op.FUENTE = viz_map.FUENTE = "http"
    op.INICIO_DATOS = pd.Timestamp("1900-01-01")       # con el barrido, first_seen es la publicación real
    venta = op.cargar(ciudad)
    if not fecha:                       # el último barrido (igual para todas las ciudades)
        from src.consolidar import rutas
        meta = rutas("venta").meta
        fecha = json.loads(meta.read_text())["ultimo_barrido"] if meta.exists() else None
    hasta = pd.Timestamp(fecha) if fecha else venta["first_seen"].max().normalize()
    desde = hasta - pd.Timedelta(days=dias - 1)
    venta["dias_pub"] = (hasta - venta["first_seen"]).dt.days

    # Oportunidades: el mismo cálculo de la alerta, con sus comparables, el otro portal y el entorno
    hist = op.historial(hasta)
    avisos, inm, _ = unificar.unificar(ciudad)
    # Un solo mercado (E8.1–E8.3): las oportunidades, las bajadas y las de captar, también de los inmuebles que solo
    # están en Metrocuadrado, comparadas con los similares de los dos portales (`pool`, sin gemelos); los movimientos
    # de la semana, por inmueble en los dos portales (src/movimientos.py).
    pool, venta, m2 = op.mercado(ciudad, venta, avisos)
    # E10: las oportunidades, las bajadas y las de captar, también de Jamundí y Yumbo; las cifras de la ciudad (pulso,
    # comunas, barrios, arriendos), solo de la ciudad
    venta["dias_pub"] = (hasta - venta["first_seen"]).dt.days        # también los de Jamundí y Yumbo
    venta_ciudad = venta[~venta["comuna"].isin(zonas.DE_EXTENSION)]
    nuevos = venta[venta["first_seen"].between(desde, hasta)]
    a_evaluar = pd.concat([nuevos] + ([m2[m2["first_seen"].between(desde, hasta)]] if len(m2) else []),
                          ignore_index=True)
    ev = op.calificar(pool, a_evaluar, hist) if len(a_evaluar) else a_evaluar
    bj = pd.concat([op.bajadas(venta, hist, desde, hasta, pool=pool)]
                   + ([op.bajadas(m2, hist, desde, hasta, pool=pool)] if len(m2) else []), ignore_index=True)
    mov = movimientos.periodo(ciudad, desde, hasta).loc["total"]
    arr = rentabilidad.arriendos(ciudad)
    ev = rentabilidad.estimar(ev, arr) if len(ev) else ev
    bj = rentabilidad.estimar(bj, arr) if len(bj) else bj
    sel = op.seleccionar(ev, UMBRAL, TOP) if len(ev) else ev
    sel_bj = op.seleccionar_bajadas(bj, sel.get("id_inmueble", pd.Index([])), TOP_BAJADAS) if len(bj) else bj
    sel = op.enriquecer(sel, pool, hasta, avisos)           # modelo, 3 similares, el otro portal, republicación
    sel_bj = op.enriquecer(sel_bj, pool, hasta, avisos)
    if len(sel):
        sel = sel.join(_entorno(sel))

    # Para captar: llevan tiempo y piden bastante más que sus comparables
    viejas = venta[venta["dias_pub"].between(*CAPTAR_DIAS)]
    cap = op.calificar(pool, viejas, hist) if len(viejas) else viejas
    captar_comunas = pd.DataFrame(columns=["viejas", "sobre"])
    if len(cap):
        cap = cap[cap["descuento"].between(-CAPTAR_SOBRE[1], -CAPTAR_SOBRE[0]) & cap["motivo_descarte"].isna()
                  & (cap["confianza"] != "Baja")]
        # En agregado (para lo público): de las que llevan 4–12 meses, cuántas piden de más, por comuna
        captar_comunas = pd.DataFrame({"viejas": viejas.groupby("comuna").size(),
                                       "sobre": cap.groupby("comuna").size()}).fillna(0).astype(int)
        cap = cap.sort_values("dias_pub", ascending=False).head(CAPTAR_FILAS).assign(n_total=len(cap))
        cap["quien"] = quien_publica(cap, avisos)
        cap = op.con_modelo(cap)                         # estimado de un modelo que no las vio

    # Oferta sin repetidos (los dos portales), con su comuna y un precio por m² saneado como el de venta
    con_zonas = zonas.tiene_zonas(ciudad)
    inm = inm.rename(columns={"latitud": "Latitud", "longitud": "Longitud", "barrio": "Barrio"})
    inm = zonas.asignar_zonas(inm, ciudad) if con_zonas else inm.assign(barrio_oficial=None, comuna=None)
    inm["precio_m2"] = inm["precio_mediano"] / inm["m2"].where(inm["m2"] > 10)
    lo, hi = venta_ciudad["precio_m2"].quantile([0.01, 0.99])
    con_precio = inm[inm["precio_m2"].between(lo, hi) & (inm["precio_mediano"] > 1e7)]
    arr_ok = arr[arr["motivo_exclusion"].isna() & ~arr["comuna"].isin(zonas.DE_EXTENSION)]
    todas = _por_comuna(con_precio, arr_ok, venta_ciudad) if con_zonas else pd.DataFrame()      # el mapa: todas
    b = con_precio.dropna(subset=["barrio_oficial"]).groupby(["comuna", "barrio_oficial"])["precio_m2"].agg(["size", "median"])
    barrios = b.loc[b["size"] >= MIN_BARRIO, "median"]                                  # el mapa por barrio
    tabla = todas.nlargest(MAX_FILAS, "viviendas").sort_values("precio_m2", ascending=False) if len(todas) else todas
    config = zonas.CIUDADES.get(ciudad, {})

    precio_m2 = con_precio["precio_m2"].median()
    canon_m2 = arr_ok["precio_m2"].median()
    return {
        "ciudad": ciudad, "desde": desde, "hasta": hasta, "con_zonas": con_zonas,
        "division": zonas.division(ciudad), "nombres": zonas.nombres(ciudad) if con_zonas else {},
        "transporte": config.get("transporte", "transporte masivo"), "fuente_zonas": config.get("fuente"),
        "viviendas": len(inm), "avisos_fr": int((avisos["portal"] == "fincaraiz").sum()),
        "avisos_m2": int((avisos["portal"] == "metrocuadrado").sum()),
        "solo_m2": int((inm["portales"] == "metrocuadrado").sum()),
        "en_los_dos": int((inm["portales"] == "fincaraiz+metrocuadrado").sum()),
        "publicadas": int(mov["publicadas"]), "retiradas": int(mov["retiradas"]),
        "bajadas": int(mov["bajadas"]),
        "bajada_mediana": None if pd.isna(mov["bajada_mediana"]) else float(mov["bajada_mediana"]),
        "arriendos": len(arr_ok), "canon_mediano": float(arr_ok["Precio"].median()),
        "precio_m2": float(precio_m2), "rentab": float(12 * canon_m2 / precio_m2),
        "tabla": tabla, "todas": todas, "barrios": barrios, "puntos": con_precio[["Latitud", "Longitud", "comuna"]],
        "evaluadas": len(ev), "con_comparables": int(ev["nivel_comps"].notna().sum()) if len(ev) else 0,
        "bajo_umbral": int((ev["descuento"] >= UMBRAL).sum()) if len(ev) else 0,
        "sel": sel, "sel_bj": sel_bj,
        "captar_n": int(cap["n_total"].iloc[0]) if len(cap) else 0, "captar": cap,
        "captar_comunas": captar_comunas, "viejas": len(viejas),
        # Para las cifras por barrio de src/contenido.py
        "inmuebles": con_precio, "venta": venta_ciudad, "arriendos_ok": arr_ok,
        # Jamundí y Yumbo (E10): sus movimientos de la semana, aparte
        "extension": extension(ciudad, desde, hasta),
    }


def extension(ciudad: str, desde: pd.Timestamp, hasta: pd.Timestamp) -> dict:
    """Los movimientos de la semana de los municipios que van con la ciudad (E10): {municipio: {en_venta,
    publicadas, retiradas, bajadas}}, por inmueble en los dos portales (src/movimientos.py)."""
    salida = {}
    for m in zonas.EXTENSION.get(ciudad, []):
        try:
            x = movimientos.periodo(m, desde, hasta).loc["total"]
        except Exception as e:             # noqa: BLE001 — sin el municipio, el informe sale igual
            print(f"⚠️  Sin los movimientos de {m}: {type(e).__name__}: {e}")
            continue
        salida[m] = {k: int(x[k]) for k in ("en_venta", "publicadas", "retiradas", "bajadas")}
    return salida


# ── Presentación ──────────────────────────────────────────────────────────────

def _n(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def _pct(x: float, dec: int = 0) -> str:
    return f"{x * 100:.{dec}f}%".replace(".", ",")


def _mill(x: float, dec: int = 1) -> str:
    """Millones de pesos: 4,5 M; los grandes sin decimales."""
    return (f"{x / 1e6:,.0f}" if x >= 1e8 else f"{x / 1e6:.{dec}f}").replace(",", "X").replace(".", ",") \
        .replace("X", ".") + " M"


def _miles(x: float) -> str:
    return f"{x / 1e3:,.0f}".replace(",", ".") + " mil"


def _fecha(d: pd.Timestamp, año: bool = True) -> str:
    return f"{d.day} de {MESES[d.month - 1]}" + (f" de {d.year}" if año else "")


def _periodo(desde: pd.Timestamp, hasta: pd.Timestamp) -> str:
    inicio = (str(desde.day) if (desde.month, desde.year) == (hasta.month, hasta.year)
              else _fecha(desde, año=desde.year != hasta.year))
    return f"Semana del {inicio} al {_fecha(hasta)}"


def _plural(division: str) -> str:
    return division + ("es" if division.endswith("d") else "s")


def _barrio(r: pd.Series, nombres: dict | None = None) -> str:
    """El barrio oficial; si no hay, el primero del aviso (algunos anunciantes listan todos los conjuntos)."""
    b = r["barrio_oficial"] if pd.notna(r.get("barrio_oficial")) else str(r["Barrio"]).split(",")[0].strip()
    b = PREFIJO.sub("", b)
    partes = [b[:1].upper() + b[1:]]
    if pd.notna(r.get("Estrato")):
        partes.append(f"estrato {r['Estrato']:.0f}")
    if nombres and pd.notna(r.get("comuna")) and r["comuna"] in nombres:
        partes.append(nombres[r["comuna"]])
    elif r.get("comuna") in zonas.DE_EXTENSION:        # Jamundí, Yumbo (E10)
        partes.append(r["comuna"])
    return " · ".join(partes)


def _dist(km: float) -> str:
    return f"{km * 1000:.0f} m" if km < 1 else f"{km:.1f} km".replace(".", ",")


def _txt(x: float, y: float, s: str, size: float = 6.6, color: str = "#0b0b0b", weight: int = 600) -> str:
    """Texto de SVG con todo en atributos: WeasyPrint no aplica el CSS de la página dentro del SVG."""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, DejaVu Sans, sans-serif" font-size="{size}pt" '
            f'font-weight="{weight}" fill="{color}" text-anchor="middle">{s}</text>')


def _encuadre(feats: list, valores: pd.Series, puntos: pd.DataFrame, ciudad: str) -> tuple[float, ...]:
    """La caja del mapa: la ciudad entera (en un área metropolitana, su núcleo) sin que la distorsionen las zonas
    grandes de alrededor. Cada comuna o localidad con datos entra entera; las enormes (más de 3 veces la mediana:
    corregimientos de Medellín, Usme o Ciudad Bolívar con su parte rural), solo por donde están sus avisos.
    Sumapaz, sin avisos, y los municipios vecinos quedan fuera (siguen en la tabla)."""
    del_nucleo = {f["properties"]["codigo"] for f in zonas.cargar("comunas", zonas.nucleo(ciudad))["features"]}
    zs = [(c, g) for c, g in feats if c in valores.index and c in del_nucleo] or \
         [(c, g) for c, g in feats if c in valores.index]
    mediana = np.median([g.area for _, g in zs])
    cajas = []
    for c, g in zs:
        p = puntos[puntos["comuna"] == c]
        if g.area <= 3 * mediana or len(p) < 10:
            cajas.append(g.bounds)
        else:
            lat, lon = (pd.to_numeric(p[k], errors="coerce").dropna() for k in ("Latitud", "Longitud"))
            cajas.append((lon.quantile(.05), lat.quantile(.05), lon.quantile(.95), lat.quantile(.95)))
    x0, y0 = min(b[0] for b in cajas), min(b[1] for b in cajas)
    x1, y1 = max(b[2] for b in cajas), max(b[3] for b in cajas)
    mx, my = (x1 - x0) * .03, (y1 - y0) * .03
    return x0 - mx, y0 - my, x1 + mx, y1 + my


def mapa(valores: pd.Series, ciudad: str, puntos: pd.DataFrame, barrios: pd.Series | None = None,
         w: int = 300, h: int = 400) -> tuple[str, str]:
    """(SVG coloreado en 5 quintiles, leyenda HTML), con el encuadre de `_encuadre` (por comunas o localidades).
    Con `barrios` (precio mediano por m² de cada barrio con datos, índice (comuna, barrio)), el color es por
    barrio y las comunas van como líneas con su número encima; sin barrios, el color es por comuna (`valores`).
    En un área metropolitana, lo que está fuera del núcleo (los municipios vecinos) va en gris:
    sus precios están en la tabla y en color confundirían. Solo llevan número las zonas donde cabe."""
    import shapely
    from shapely.geometry import shape
    feats = [(f["properties"]["codigo"], shape(f["geometry"])) for f in zonas.cargar("comunas", ciudad)["features"]]
    etiquetas = zonas.etiquetas(ciudad)
    x0, y0, x1, y1 = _encuadre(feats, valores, puntos, ciudad)
    area = ciudad in zonas.AREAS
    del_nucleo = {f["properties"]["codigo"] for f in zonas.cargar("comunas", zonas.nucleo(ciudad))["features"]}
    cl = np.cos(np.radians((y0 + y1) / 2))
    k = min((w - 8) / ((x1 - x0) * cl), (h - 8) / (y1 - y0))
    punto = lambda x, y: (4 + (x - x0) * cl * k, 4 + (y1 - y) * k)
    por_barrio = barrios is not None and len(barrios) >= 10
    base = barrios if por_barrio else valores
    cortes = base.quantile([.2, .4, .6, .8]).to_numpy()
    clase = lambda v: int(np.searchsorted(cortes, v))

    def camino(g) -> str:
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        return " ".join("M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in (punto(*c) for c in p.exterior.coords)) + "Z"
                        for p in polys)
    fondo, lineas, textos = [], [], []
    if por_barrio:
        for f in zonas.cargar("barrios", ciudad)["features"]:
            v = barrios.get((f["properties"].get("comuna"), f["properties"]["nombre"]))
            fondo.append(f'<path d="{camino(shape(f["geometry"]))}" fill="{RAMPA[clase(v)] if pd.notna(v) else "#f0efec"}" '
                         f'stroke="#ffffff" stroke-width="0.4"/>')
    for cod, g in feats:
        fuera = area and cod not in del_nucleo
        v = None if fuera else valores.get(cod)
        if por_barrio:
            if fuera:
                fondo.insert(0, f'<path d="{camino(g)}" fill="{ALREDEDORES}" stroke="#ffffff" stroke-width="1"/>')
            else:
                lineas.append(f'<path d="{camino(g)}" fill="none" stroke="#52514e" stroke-width="0.9"/>')
        else:
            relleno = ALREDEDORES if fuera else (RAMPA[clase(v)] if pd.notna(v) else "#f0efec")
            fondo.append(f'<path d="{camino(g)}" fill="{relleno}" stroke="#ffffff" stroke-width="1"/>')
        visible = shapely.clip_by_rect(g, x0, y0, x1, y1)
        if visible.is_empty:
            continue
        cx, cy = punto(*visible.representative_point().coords[0])
        bx0, by1, bx1, by0 = visible.bounds
        ancho, alto = (bx1 - bx0) * cl * k, (by0 - by1) * k
        if etiquetas.get(cod) and 4 <= cx <= w - 4 and 6 <= cy <= h - 4 and min(ancho, alto) >= 16:
            if por_barrio:                  # sobre barrios de colores: el número en un círculo blanco
                textos.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="#ffffff" fill-opacity="0.85"/>')
                textos.append(_txt(cx, cy + 2.2, etiquetas[cod], size=5.8))
            else:
                textos.append(_txt(cx, cy + 2.5, etiquetas[cod], color="#ffffff" if pd.notna(v) and clase(v) >= 3 else "#0b0b0b"))
    lims = [base.min(), *cortes, base.max()]
    leyenda = "".join(f'<span class="ley"><i style="background:{c}"></i>{_mill(lims[i])}–{_mill(lims[i + 1])}</span>'
                      for i, c in enumerate(RAMPA)) + '<span class="ley"><i style="background:#f0efec"></i>pocos datos</span>'
    if por_barrio:
        leyenda += f'<span class="ley"><i class="linea"></i>{_plural(zonas.division(ciudad))}</span>'
    if area:
        leyenda += f'<span class="ley"><i style="background:{ALREDEDORES}"></i>alrededores: ver la tabla</span>'
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{"".join(fondo + lineas + textos)}</svg>', leyenda


def frases(d: dict) -> list[str]:
    t, nombres = d["tabla"], d["nombres"]
    lineas = [
        f"En {d['ciudad']} hay <b>{_n(d['viviendas'])} viviendas en venta</b> sin contar repetidos: una vivienda "
        f"publicada en varios portales o por varias inmobiliarias cuenta una vez ({_n(d['avisos_fr'] + d['avisos_m2'])} "
        f"avisos en total).",
        f"Esta semana se publicaron <b>{_n(d['publicadas'])}</b> viviendas, se confirmó el retiro de "
        f"<b>{_n(d['retiradas'])}</b> y <b>{_n(d['bajadas'])}</b> bajaron de precio"
        + (f" (rebaja promedio del {_pct(d['bajada_mediana'])})." if d["bajada_mediana"] else "."),
    ]
    if ext := d.get("extension"):           # E10: Jamundí y Yumbo, aparte de las cifras de la ciudad
        lineas.append("Con " + d["ciudad"] + " van " + " y ".join(ext) + ": "
                      + "; ".join(f"en {m}, <b>{_n(x['en_venta'])}</b> en venta, {_n(x['publicadas'])} publicadas y "
                                  f"{_n(x['bajadas'])} bajaron" for m, x in ext.items())
                      + ". Sus oportunidades, bajadas y viviendas para captar salen con las de " + d["ciudad"] + ".")
    b = d.get("barrio_semana")
    if not d.get("mensual", True) and b:
        lineas.append(f"El barrio de la semana es <b>{html.escape(PREFIJO.sub('', b['barrio']))}</b> "
                      f"({html.escape(b['comuna_nombre'])}): {_mill(b['precio_m2'])}/m², {_puesto(b)}.")
    elif len(t):
        cara, rent = t.index[0], t["rentab"].dropna()
        lineas.append(
            f"La zona más cara por m² es <b>{html.escape(nombres[cara])}</b> "
            f"({html.escape(str(t.loc[cara, 'barrios']))}), con {_mill(t.loc[cara, 'precio_m2'])}/m²"
            + (f"; la de mayor rentabilidad bruta típica, <b>{html.escape(nombres[rent.idxmax()])}</b> "
               f"({html.escape(str(t.loc[rent.idxmax(), 'barrios']))}), con {_pct(rent.max(), 1)}." if len(rent) else "."))
    return lineas


def _tile(etiqueta: str, valor: str, nota: str = "") -> str:
    return (f'<div class="tile"><div class="t-label">{etiqueta}</div><div class="t-value">{valor}</div>'
            f'<div class="t-note">{nota}</div></div>')


def _fila(nombre: str, r: pd.Series, pmax: float, rmax: float) -> str:
    barra = lambda v, m, clase: (f'<span class="bar {clase}" style="width:{v / m * 100:.1f}%"></span>'
                                 if pd.notna(v) else "")
    canon = _miles(r["canon_m2"]) if pd.notna(r["canon_m2"]) else "—"
    rent = _pct(r["rentab"], 1) if pd.notna(r["rentab"]) else "—"
    parte = r.get("en_venta_parte")
    del_total = _pct(parte, 1) if parte is not None and pd.notna(parte) else "—"
    return f"""
      <tr>
        <td class="c-name"><b>{html.escape(nombre)}</b> <span>{html.escape(str(r['barrios']))}</span></td>
        <td class="num">{_n(r['viviendas'])}</td>
        <td class="num">{del_total}</td>
        <td class="num">{_mill(r['precio_m2'])}</td>
        <td class="bar-cell">{barra(r['precio_m2'], pmax, 'b-precio')}</td>
        <td class="num">{r['dias_pub']:.0f}</td>
        <td class="num">{canon}</td>
        <td class="num">{rent}</td>
        <td class="bar-cell narrow">{barra(r['rentab'], rmax, 'b-rentab')}</td>
      </tr>"""


def _frente_modelo(r: pd.Series) -> str:
    """'18% más' / '5% menos' / 'en línea' frente al estimado del modelo (que no vio el aviso); '—' sin él."""
    b = r.get("bajo_modelo")
    if pd.isna(b):
        return "—"
    return "en línea" if abs(b) < .03 else f"{_pct(abs(b))} {'menos' if b > 0 else 'más'}"


def _credito_html(d: dict) -> str:
    """El crédito de vivienda en la primera página (src/credito.py): la tasa media en pesos y en UVR de las últimas
    4 semanas, los créditos para comprar vivienda (solo si el volumen es fiable) y la cuota del apartamento típico.
    Vacío sin la serie."""
    c = d.get("credito")
    if not c or not c.get("tasa_pesos"):
        return ""
    año = (c.get("hace_un_año") or {}).get("tasa_pesos")
    ej = c["ciudad_ejemplo"]
    al = credito.fecha_corta(c["hasta"])
    cambio = (c.get("pais") or {}).get("cambio_año")
    pais = (f" · país: {_signo(cambio, 0)} frente a hace un año" if cambio is not None else "")
    return f"""<div class="tiles">
  {_tile("Tasa del crédito de vivienda", credito.pct(c['tasa_pesos']), f"efectiva anual (E.A.), en pesos; promedio de 4 semanas al {al}" + (f" · hace un año {credito.pct(año)}" if año else ""))}
  {_tile("En UVR", credito.pct(c['tasa_uvr']) + " + UVR" if c.get('tasa_uvr') else "—", "E.A.; la cuota sube con la inflación")}
  {_tile("Créditos para comprar", _n(c['creditos']) if c.get('fiable') else "—", (f"en {d['ciudad']}, 4 semanas" if c.get('fiable') else "esta vez el dato llegó incompleto") + pais)}
  {_tile("Cuota del apartamento típico", credito.millones(ej['cuota']) + "/mes", ("en pesos" + (f"; en UVR, desde {credito.millones(ej['cuota_uvr'])}" if ej.get('cuota_uvr') else "") + f" · {_n(ej['area'])} m² a {_mill(ej['precio'], 0)}, {credito.FINANCIADO * 100:.0f} % a {credito.AÑOS} años").replace(" M", "&nbsp;M").replace(" %", "&nbsp;%").replace(" años", "&nbsp;años"))}
</div>"""


def _oportunidad(r: pd.Series, i: int, d: dict) -> str:
    hasta = d["hasta"]
    dias = (hasta - r["first_seen"]).days
    lineas = [f"<b>{r['n_comps']} similares</b> {'en el barrio' if r['nivel_comps'] == 'barrio' else 'en la zona'}, "
              f"promedio {_mill(r['med_comps'])}/m² · confianza {r['confianza'].lower()}"]
    millones = lambda x: _mill(x, 0)
    if modelo := op._modelo_txt(r, millones):
        lineas.append(f"<b>Modelo de precios:</b> {html.escape(modelo.removeprefix('modelo: '))}")
    if cuota := credito.texto_cuota(r["Precio"], d["ciudad"]):
        lineas.append(f"<b>{html.escape(cuota.split(':')[0])}:</b>{html.escape(cuota.split(':', 1)[1])}")
    if rep := op._republicado_txt(r, hasta, millones):
        lineas.append(html.escape(op._mayus(rep)))
    if pd.notna(r.get("m2_precio")):
        dif = r["m2_precio"] / r["Precio"] - 1
        cuanto = "mismo precio" if abs(dif) < .005 else f"{abs(dif):.0%} {'más' if dif > 0 else 'menos'}"
        lineas.append(f'<b>También en otro portal:</b> {_mill(r["m2_precio"], 0)} ({cuanto}) · '
                      f'<a href="{html.escape(str(r["m2_url"]))}">ver ↗</a>')
    if pd.notna(r.get("canon_est")):
        lineas.append(f"Arriendo estimado {_mill(r['canon_est'])}/mes · rentabilidad bruta {_pct(r['rentab_bruta'], 1)} "
                      f"({r['n_arr']:.0f} similares)")
    if pd.notna(r.get("dist_transporte_masivo_km")):
        # Solo distancias: los conteos de OSM (comercios, colegios a 500 m) salen en 0 donde el mapa está incompleto
        # (28% y 69% de los avisos de Cali, 7-nov), y un «0 comercios» engaña
        lineas.append(f"<b>Entorno:</b> {d['transporte']} {_dist(r['dist_transporte_masivo_km'])} · parque "
                      f"{_dist(r['dist_parque_km'])} · c. comercial {_dist(r['dist_centro_comercial_km'])}")
    comps = "".join(f"<tr><td>{html.escape(PREFIJO.sub('', str(c['barrio_oficial'] or c['Barrio'])))}</td>"
                    f"<td class='num'>{c['Area_m2']:.0f} m²</td><td class='num'>{_mill(c['Precio'], 0)}</td>"
                    f"<td class='num'>{_mill(c['precio_m2'])}/m²</td>"
                    f"<td><a href='{html.escape(str(c['URL detalle']))}'>ver ↗</a></td></tr>" for _, c in r["comps"].iterrows())
    return f"""
  <article class="opo">
    <div class="opo-top"><a class="ver" href="{html.escape(str(r['URL detalle']))}">Ver aviso ↗</a>
      <span class="badge ok">{_pct(r['descuento'])} bajo similares</span>{
      ' <span class="badge ojo">verificar área y precio</span>' if r["verificar"] else ''}</div>
    <div class="opo-cuerpo">
      <div>
        <h3>{i}. {html.escape(op._ficha(r))}</h3>
        <p class="zona">{html.escape(_barrio(r, d['nombres']))}</p>
        <p class="precio"><b>{_mill(r['Precio'], 0)}</b> · {_mill(r['precio_m2'])}/m² · publicado hace {dias} día{'s' if dias != 1 else ''}</p>
        {''.join(f'<p class="hecho">{l}</p>' for l in lineas)}
      </div>
      <div><p class="mini-tit">3 de sus similares típicos, los más parecidos</p>
        <table class="mini">{comps}</table></div>
    </div>
  </article>"""


def _por_zona_html(d: dict, periodo: str) -> str:
    """El mapa (página 1) y la tabla por comuna o localidad (página 2)."""
    t, nombres, div = d["tabla"], d["nombres"], d["division"]
    filas = "".join(_fila(nombres.get(c, c), r, t["precio_m2"].max(), t["rentab"].max()) for c, r in t.iterrows())
    svg_mapa, leyenda = mapa(d["todas"]["precio_m2"], d["ciudad"], d["puntos"], d["barrios"])
    lista = lambda cs: ", ".join(html.escape(nombres.get(c, c)) for c in cs)
    Div, divs = div.capitalize(), _plural(div)
    por_barrio = len(d["barrios"]) >= 10
    aproximados = por_barrio and any(f["properties"].get("aproximado") for f in zonas.cargar("barrios", d["ciudad"])["features"])
    nota_mapa = (f"<p>Cada barrio con al menos {MIN_BARRIO} viviendas en venta lleva su color; las líneas son las "
                 f"{divs}.{' Barrios aproximados a partir de OpenStreetMap.' if aproximados else ''}</p>"
                 if por_barrio else "")
    from src.mapa_ciudad import URL as URL_MAPAS
    enlace = (f'<p class="enlace"><a href="{URL_MAPAS}/{zonas.slug(d["ciudad"])}.html">Ver mapa interactivo ↗</a> '
              f'todos los avisos, por barrio y con su enlace</p>')
    return f"""<h2>Precio por m² por {'barrio' if por_barrio else div}</h2>
<div class="mapa-fila"><div>{svg_mapa}</div>
  <div class="mapa-txt">{enlace}<div class="leyenda">{leyenda}</div>{nota_mapa}
    <p>Las zonas más caras por m²: {lista(t.index[:4])}. Las de mayor rentabilidad bruta típica: {lista(t["rentab"].nlargest(3).index)}.</p>
    <p>El detalle de cada {div}, con los días que llevan publicados sus avisos, el canon y la rentabilidad, en la
      página siguiente.</p></div></div>

<div class="salto"></div>
<p class="kicker">Informe semanal · {d['ciudad']}</p>
<h1>Mercado por {div}</h1>
<p class="periodo">{periodo} · de la más cara a la más económica por m² · {divs} con al menos {MIN_VIVIENDAS} viviendas en venta</p>
<table>
  <thead><tr><th>{Div} y barrios con más oferta</th><th class="num">Viviendas<br>en venta</th>
    <th class="num">Del total<br>(catastro)</th><th class="num">Precio promedio<br>por m²</th><th></th><th class="num">Días<br>publicados</th>
    <th class="num">Canon promedio<br>por m²</th><th class="num">Rentabilidad<br>bruta típica</th><th></th></tr></thead>
  <tbody>{filas}</tbody>
</table>
<p class="nota">Del total: las viviendas en venta frente a las que hay en el catastro de Cali (IDESC); «—» si más
  de un cuarto de lo que está en venta trae una ubicación genérica (el centro del barrio o de la ciudad). Donde se
  construye, cuenta también lo nuevo que aún no está en el catastro.
  Días publicados: edad promedio de los avisos activos de la {div} (desde su fecha de publicación).
  Rentabilidad bruta típica: canon promedio por m² × 12 ÷ precio promedio por m², antes de administración, impuestos y
  vacancia; «—»: menos de {MIN_ARRIENDOS} arriendos.</p>"""


def _puesto(b: dict) -> str:
    p, de = b["puesto_m2"], b["de"]
    if p == 1:
        return f"el más caro de los {de} barrios con al menos 15 viviendas en venta"
    if p == de:
        return f"el más económico de los {de} barrios con al menos 15 viviendas en venta"
    return (f"el {p}.º más caro" if p <= de / 2 else f"el {de - p + 1}.º más económico") + \
        f" de {de} barrios con al menos 15 viviendas en venta"


def _signo(x: float, dec: int = 1) -> str:
    """Con signo (+0,4 %, −1,2 %); lo que redondea a cero, sin signo."""
    v = round(x * 100, dec) + 0.0
    return (f"{v:.{dec}f}%" if v == 0 else f"{v:+.{dec}f}%").replace(".", ",").replace("-", "−")


def uno_de_cada(parte: float) -> str:
    """0,203 → '1 de cada 5'."""
    return f"1 de cada {_n(round(1 / parte))}"


def _catastro_barrio(b: dict, ciudad: str) -> str:
    """Lo que dice el catastro del barrio (src/catastro_cali.py): cuántas viviendas hay, cuántas son apartamentos y
    en edificios de cuántos pisos, y qué parte está en venta (solo con pocas coordenadas genéricas). Vacío sin él."""
    if not b.get("viviendas_catastro"):
        return ""
    t = f"<b>En el catastro:</b> {_n(b['viviendas_catastro'])} viviendas"
    if (ap := b.get("apartamentos_parte_catastro")) is not None and ap >= 0.05:
        t += f", el {_pct(ap)} apartamentos" + (f" en edificios de unos {b['pisos_edificio']} pisos" if b.get("pisos_edificio") else "")
        t += f" ({_n(b['edificios'])} edificios o conjuntos)" if b.get("edificios") else ""
    t += "."
    if b.get("en_venta_parte"):
        t += (f" Hoy está en venta el equivalente a <b>{uno_de_cada(b['en_venta_parte'])}</b>"
              + (f" ({ciudad}: {uno_de_cada(b['ciudad_en_venta_parte'])})" if b.get("ciudad_en_venta_parte") else "")
              + "; donde se construye, cuenta también lo nuevo que aún no está en el catastro.")
    for tipo, plural in (("apartamento", "apartamentos"), ("casa", "casas")):      # E9.4: el avalúo catastral
        if b.get(f"avaluo_m2_{tipo}") and b.get(f"veces_avaluo_{tipo}"):
            cali = b.get(f"ciudad_veces_avaluo_{tipo}")
            t += (f" <b>Avalúo catastral:</b> {_mill(b[f'avaluo_m2_{tipo}'], 2)}/m² construido en sus {plural}; se piden "
                  f"<b>{str(b[f'veces_avaluo_{tipo}']).replace('.', ',')} veces</b> su avalúo"
                  + (f" ({ciudad}: {str(cali).replace('.', ',')})" if cali else "") + ".")
            break
    return f"<p>{t}</p>"


def _barrio_html(d: dict) -> str:
    """Las semanas que no son la del mes: un barrio en lugar del mapa y la tabla por comuna, que de una semana a otra
    casi no cambian (src/contenido.py elige el barrio y rota entre todos)."""
    b, div = d.get("barrio_semana"), d["division"]
    aviso = (f"El mapa y la tabla por {div} salen en el primer informe de cada mes: de una semana a otra apenas "
             f"cambian.")
    if not b:
        return f'<p class="nota">{aviso}</p>'
    nom, com = html.escape(PREFIJO.sub("", b["barrio"])), html.escape(b["comuna_nombre"])
    h = b.get("habitaciones")
    hab = "" if h is None else (f"{h:.0f}" if float(h).is_integer() else f"{int(h)}–{int(h) + 1}") + " habitaciones · "
    bajo = b["bajaron_mes"] / b["en_venta_mes"] if b.get("en_venta_mes") else None
    renta = (f"Rentabilidad bruta típica: <b>{_pct(b['rentabilidad'], 1)}</b> (canon promedio {_miles(b['canon_m2'])}/m² "
             f"con {_n(b['arriendos'])} arriendos; {d['ciudad']}: {_pct(b['ciudad_rentabilidad'], 1)})."
             if b.get("rentabilidad") else
             f"Hay menos de 15 arriendos publicados en el barrio: sin rentabilidad propia"
             + (f" (la de su {div}: {_pct(b['comuna_rentabilidad'], 1)})." if b.get("comuna_rentabilidad") else "."))
    filas = "".join(
        f"<tr><td>{'<b>' if x['barrio'] == b['barrio'] else ''}{html.escape(PREFIJO.sub('', x['barrio']))}"
        f"{'</b>' if x['barrio'] == b['barrio'] else ''}</td><td class='num'>{_n(x['viviendas'])}</td>"
        f"<td class='num'>{_mill(x['precio_m2'])}</td></tr>" for x in b.get("su_comuna") or [])
    tabla = (f'<p class="mini-tit">Los barrios de la {com} con al menos 15 viviendas en venta</p>'
             f'<table class="mini"><thead><tr><th>Barrio</th><th class="num">Viviendas en venta</th>'
             f'<th class="num">Precio promedio por m²</th></tr></thead><tbody>{filas}</tbody></table>'
             if len(b.get("su_comuna") or []) >= 2 else "")
    conj = b.get("conjuntos") or []
    if conj:                                # para captar: los conjuntos más grandes (catastro), solo en el informe
        filas_c = "".join(f"<tr><td>{html.escape(x['direccion'] or '—')}</td><td class='num'>{_n(x['apartamentos'])}</td>"
                          f"<td class='num'>{x['pisos'] or '—'}</td><td class='num'>{_n(x['area'])} m²</td></tr>"
                          for x in conj)
        tabla_c = (f'<p class="mini-tit">Los edificios y conjuntos con más apartamentos (catastro), para buscar captaciones</p>'
                   f'<table class="mini"><thead><tr><th>Dirección</th><th class="num">Apartamentos</th>'
                   f'<th class="num">Pisos</th><th class="num">Área típica</th></tr></thead><tbody>{filas_c}</tbody></table>')
        tabla = f'<div class="dos">{f"<div>{tabla}</div>" if tabla else ""}<div>{tabla_c}</div></div>'
    return f"""<h2>El barrio de la semana: {nom}</h2>
<p class="sub">{com} · {_n(b['viviendas'])} viviendas en venta · {_puesto(b)} por precio del m²</p>
<div class="tiles">
  {_tile("Precio promedio por m²", _mill(b['precio_m2']), f"{com}: {_mill(b['comuna_precio_m2']) if b.get('comuna_precio_m2') else '—'} · {d['ciudad']}: {_mill(b['ciudad_precio_m2'])}")}
  {_tile("Vivienda típica", f"{_n(b['area'])} m²", f"{hab}{_mill(b['precio'])} · la mitad entre {_mill(b['precio_p25'])} y {_mill(b['precio_p75'])}")}
  {_tile("Llevan publicadas", f"{_n(b['dias_publicadas'])} días" if b.get('dias_publicadas') is not None else "—", "promedio de las que están en venta")}
  {_tile("Bajaron de precio", _pct(bajo, 1) if bajo is not None else "—", f"en los últimos {b['dias_bajadas']} días")}
</div>
<p>{renta} El estrato más común es el {b.get('estrato') or '—'}; el {_pct(b['apartamentos'])} son apartamentos.</p>
{_catastro_barrio(b, d['ciudad'])}
{tabla}
<p class="nota">Cada semana, un barrio distinto. {aviso}</p>"""


def _credito_mes_html(d: dict) -> str:
    """El crédito de vivienda del último mes completo: la tasa frente al mes anterior y al mismo mes del año pasado;
    el volumen, sin comparar (el de la ciudad no se compara con el año pasado: src/credito.py) y solo si es fiable."""
    m = (d.get("credito") or {}).get("mes")
    if not m or not m.get("tasa_pesos"):
        return ""
    pct = credito.pct
    comparar = [f"{x['nombre']}{'' if x['año'] == m['año'] else ' de ' + str(x['año'])}: {pct(x['tasa_pesos'])}"
                for x in (m.get("anterior"), m.get("hace_un_año")) if x and x.get("tasa_pesos")]
    texto = (f"<b>El crédito de vivienda en {m['nombre']}:</b> tasa promedio del {pct(m['tasa_pesos'])} E.A. en pesos"
             + (f" ({'; '.join(comparar)})" if comparar else "")
             + (f" y {pct(m['tasa_uvr'])} E.A. + la inflación en UVR" if m.get("tasa_uvr") else ""))
    if m.get("fiable"):
        texto += f"; {_n(m['creditos'])} créditos para comprar vivienda en {d['ciudad']}"
    return f'<p class="sub" style="margin-top:8pt">{texto}.</p>'


def _mes_html(d: dict) -> str:
    """La semana del mes: lo que se movió el mes anterior (src/contenido.py: mes)."""
    m, div = d["mes"], d["division"]
    if not m["completo"]:
        return (f'<p class="nota">El resumen del mes empieza con el primer mes completo del barrido diario (desde el '
                f'25 de octubre de 2026): {m["nombre"]} no se vio entero.</p>')
    a = m.get("anterior")
    antes = lambda k, f=_n: f" · {a['nombre']}: {f(a[k])}" if a and a.get(k) is not None else ""
    fin = pd.Timestamp(m["hasta"])
    filas = "".join(
        f"<tr><td>{html.escape(c['nombre'])}</td><td class='num'>{_n(c['en_venta_todo'])}</td>"
        f"<td class='num'>{_n(c['publicadas'])}</td><td class='num'>{_n(c['retiradas'])}</td>"
        f"<td class='num'>{_pct(c['bajaron'] / c['en_venta_todo'], 1)}</td>"
        f"<td class='num'>{_pct(c['subieron'] / c['en_venta_todo'], 1)}</td>"
        f"<td class='num'>{_signo(c['cambio_mismas'])}</td></tr>"
        for c in sorted(m["comunas"], key=lambda c: -c["en_venta_todo"]))
    return f"""<div class="salto"></div>
<p class="kicker">Informe semanal · {d['ciudad']} · el mes</p>
<h1>El mes de {m['nombre']}</h1>
<p class="periodo">Del 1 al {fin.day} de {m['nombre']} de {fin.year} · lo que entró, lo que salió y cuánto cambió el precio
  de las mismas viviendas</p>
<div class="tiles">
  {_tile("Publicadas", _n(m['publicadas']), "viviendas nuevas, sin repetidos" + antes('publicadas'))}
  {_tile("Retiradas", _n(m['retiradas']), f"confirmadas tras {AUSENCIAS_PARA_BAJA} días sin verse" + antes('retiradas'))}
  {_tile("Bajaron de precio", _n(m['bajaron']), f"{_pct(m['bajaron'] / m['en_venta_todo'], 1)} de las {_n(m['en_venta_todo'])} en venta todo el mes" + antes('bajaron'))}
  {_tile("Precio de las mismas viviendas", _signo(m['cambio_mismas']), f"{_n(m['subieron'])} subieron" + (f" · bajada promedio {_pct(m['bajada_mediana'], 1)}" if m.get('bajada_mediana') else "") + antes('cambio_mismas', _signo))}
</div>
<table class="mini"><thead><tr><th>{div.capitalize()}</th><th class="num">En venta<br>todo el mes</th>
  <th class="num">Publicadas</th><th class="num">Retiradas</th><th class="num">Bajaron<br>de precio</th>
  <th class="num">Subieron<br>de precio</th><th class="num">Precio de las<br>mismas viviendas</th></tr></thead>
  <tbody>{filas}</tbody></table>
{_credito_mes_html(d)}
<p class="nota">{_plural(div).capitalize()} con al menos {m['min_comuna']} viviendas en venta todo el mes. Bajaron o subieron: un
  cambio de su precio pedido de al menos {_pct(op.BAJA_MIN)}. Precio de las mismas viviendas: el cambio promedio del precio
  pedido de las que estuvieron en venta todo el mes. No comparamos el promedio del m² con el del mes anterior: sube o
  baja sobre todo según qué viviendas entran y salen (en tres semanas de noviembre, un 15–20 % de las {_plural(div)} y
  barrios cambió más de un 3 % así, mientras el precio de las mismas viviendas no se movía).</p>"""


def informe_html(d: dict) -> str:
    t, nombres, div = d["tabla"], d["nombres"], d["division"]
    periodo = _periodo(d["desde"], d["hasta"])
    mensual = d.get("mensual", True)
    por_zona = (_por_zona_html(d, periodo) if mensual else _barrio_html(d)) if len(t) else (
        f'<p class="nota">Todavía no tenemos los polígonos oficiales de {_plural(div)} y barrios de {d["ciudad"]}: '
        f'el informe va sin mapa ni tabla por {div}, y los similares se buscan a menos de 1 km.</p>')
    del_mes = _mes_html(d) if mensual and d.get("mes") else ""
    oportunidades = "".join(_oportunidad(r, i, d) for i, (_, r) in enumerate(d["sel"].iterrows(), 1))
    bajadas = "".join(
        f"<tr><td>{html.escape(op._ficha(r))}</td><td>{html.escape(_barrio(r, nombres))}</td>"
        f"<td class='num'>{_mill(r['precio_antes'], 0)}</td><td class='num'><b>{_mill(r['Precio'], 0)}</b></td>"
        f"<td class='num'>−{_pct(r['baja'])}{'<br><span class=nota-c>al republicar</span>' if r.get('al_republicar') else ''}</td>"
        f"<td class='num'>{html.escape((op._frente_comps(r) or '—').replace('ahora ', '').replace('sus comparables', 'similares'))}</td>"
        f"<td><a href='{html.escape(str(r['URL detalle']))}'>ver ↗</a></td></tr>" for _, r in d["sel_bj"].iterrows())
    captar = "".join(
        f"<tr><td>{html.escape(_barrio(r, nombres))}</td><td class='nw'>{html.escape(r['Tipo_propiedad'])} · {r['Area_m2']:.0f} m²</td>"
        f"<td class='num'>{_mill(r['Precio'], 0)}</td><td class='num'>{_pct(-r['descuento'])} más</td>"
        f"<td class='num'>{_frente_modelo(r)}</td>"
        f"<td class='num'>{r['dias_pub']:.0f}</td><td class='nw'>{html.escape(r.get('quien') or '')}</td>"
        f"<td><a href='{html.escape(str(r['URL detalle']))}'>ver ↗</a></td></tr>"
        for _, r in d["captar"].iterrows())
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Pulso del mercado · {d['ciudad']}</title>
<style>
  @page {{ size: A4; margin: 12mm 14mm 14mm;
    @bottom-left {{ content: "{tema.prefijo()}Informe semanal · {d['ciudad']} · {_fecha(d['hasta'])}"; font: 7pt Inter, sans-serif; color: #898781; }}
    @bottom-center {{ content: "{tema.contacto_pie()}"; font: 7pt Inter, sans-serif; color: #898781; }}
    @bottom-right {{ content: "Página " counter(page) " de " counter(pages); font: 7pt Inter, sans-serif; color: #898781; }} }}
  :root {{ --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --line:#e1e0d9; --plane:#f9f9f7;
           --blue:#2a78d6; --orange:#eb6834; --good:#006300; --good-bg:#e8f3e8; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: Inter, "DejaVu Sans", sans-serif; font-size: 8.5pt; line-height: 1.45; color: var(--ink); margin: 0; }}
  .kicker {{ font-size: 7.2pt; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0 0 2pt; }}
  h1 {{ font-size: 21pt; font-weight: 600; margin: 0; letter-spacing: -.01em; }}
  .periodo {{ color: var(--ink-2); margin: 2pt 0 10pt; }}
  h2 {{ font-size: 11.5pt; font-weight: 600; margin: 12pt 0 2pt; }}
  h3 {{ font-size: 9.6pt; font-weight: 600; margin: 0; }}
  .sub {{ color: var(--ink-2); margin: 0 0 7pt; }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6pt; margin-bottom: 6pt; }}
  .tile {{ background: var(--plane); border-radius: 6pt; padding: 6pt 9pt 7pt; }}
  .t-label {{ color: var(--ink-2); font-size: 7.6pt; }}
  .t-value {{ font-size: 16pt; font-weight: 600; margin: 0; }}
  .t-note {{ color: var(--muted); font-size: 7pt; line-height: 1.3; }}
  .resumen {{ border-left: 2.5pt solid var(--blue); padding: 1pt 0 1pt 9pt; margin: 8pt 0 0; }}
  .resumen ul {{ margin: 0; padding-left: 10pt; }} .resumen li {{ margin-bottom: 2pt; }}
  .mapa-fila {{ display: grid; grid-template-columns: 300pt 1fr; gap: 14pt; align-items: start; margin-top: 4pt; }}
  .mapa-txt p {{ color: var(--ink-2); margin: 7pt 0 0; }}
  .leyenda {{ line-height: 1.9; }}
  .ley {{ display: inline-block; margin-right: 9pt; font-size: 7.2pt; color: var(--ink-2); white-space: nowrap; }}
  .ley i {{ display: inline-block; width: 9pt; height: 9pt; border-radius: 2pt; vertical-align: -1.5pt; margin-right: 3pt; }}
  .ley i.linea {{ background: none; border: 1.2pt solid #52514e; border-radius: 1pt; }}
  .enlace {{ margin: 0 0 8pt !important; color: var(--ink-2); font-size: 7.6pt; }}
  .enlace a {{ font-size: 9pt; display: block; margin-bottom: 1pt; }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th {{ text-align: left; font-weight: 500; color: var(--ink-2); font-size: 7.4pt; padding: 0 4pt 4pt;
        border-bottom: 1px solid var(--line); vertical-align: bottom; }}
  th.num, td.num {{ text-align: right; }}
  tr {{ break-inside: avoid; }}
  td {{ padding: 3pt 4pt; border-bottom: 1px solid var(--line); vertical-align: middle; }}
  td.num, td:last-child {{ white-space: nowrap; }}
  td.c-name span {{ color: var(--muted); font-size: 7.4pt; }}
  td.bar-cell {{ width: 18%; padding-left: 2pt; }} td.bar-cell.narrow {{ width: 11%; }}
  .bar {{ display: block; height: 6pt; border-radius: 0 3pt 3pt 0; }}
  .b-precio {{ background: var(--blue); }} .b-rentab {{ background: var(--orange); }}
  .nota {{ color: var(--muted); font-size: 7pt; line-height: 1.4; margin-top: 8pt; }}
  .salto {{ break-before: page; }}
  .opo {{ border: 1px solid var(--line); border-radius: 6pt; padding: 5pt 9pt 6pt; margin-bottom: 5pt; break-inside: avoid; }}
  .opo-top {{ overflow: hidden; margin-bottom: 2pt; }}   /* sin flex: WeasyPrint pierde los enlaces dentro de flex o grid */
  .opo-top .ver {{ float: right; }}
  .opo-cuerpo {{ display: grid; grid-template-columns: 1.45fr 1fr; gap: 10pt; }}
  .badge {{ font-weight: 600; font-size: 7.6pt; border-radius: 99pt; padding: 1.5pt 7pt; background: var(--good-bg); color: var(--good); }}
  .badge.ojo {{ background: #fdf1dc; color: #7a4a00; }}
  .opo .zona {{ color: var(--ink-2); margin: 1pt 0 3pt; }}
  .opo .precio {{ font-size: 9.2pt; margin: 0 0 3pt; }}
  .opo .hecho {{ margin: 0 0 1pt; color: var(--ink-2); font-size: 7.4pt; line-height: 1.32; }} .opo .hecho b {{ color: var(--ink); font-weight: 600; }}
  a {{ color: var(--blue); font-weight: 600; text-decoration: none; }}
  .nota-c {{ color: var(--muted); font-size: 6.8pt; }}
  td.nw {{ white-space: nowrap; }}
  .mini-tit {{ color: var(--ink-2); font-size: 7.4pt; margin: 0 0 2pt; }}
  table.mini td, table.mini th {{ font-size: 7.6pt; padding: 2.2pt 3pt; }}
  .dos {{ display: grid; grid-template-columns: 1fr 1.25fr; gap: 12pt; }}
  .vacio {{ color: var(--ink-2); }}
  {tema.css()}
</style></head><body>
{tema.cabecera()}

<p class="kicker">Informe semanal · vivienda en venta y arriendo</p>
<h1>Pulso del mercado · {d['ciudad']}</h1>
<p class="periodo">{periodo} · datos de los principales portales inmobiliarios</p>
<div class="tiles">
  {_tile("Viviendas en venta", _n(d['viviendas']), f"sin repetidos · {_n(d['avisos_fr'] + d['avisos_m2'])} avisos")}
  {_tile("Publicadas esta semana", _n(d['publicadas']), "viviendas nuevas, sin repetidos")}
  {_tile("Retiradas esta semana", _n(d['retiradas']), f"confirmadas tras {AUSENCIAS_PARA_BAJA} días sin verse")}
  {_tile("Bajaron de precio", _n(d['bajadas']), f"rebaja promedio del {_pct(d['bajada_mediana'])}" if d['bajada_mediana'] else "")}
</div>
<div class="tiles">
  {_tile("Precio promedio por m²", _mill(d['precio_m2']), "vivienda en venta, toda la ciudad")}
  {_tile("Arriendos de vivienda", _n(d['arriendos']), "sin amoblados ni por días")}
  {_tile("Canon promedio", _mill(d['canon_mediano']), "por mes")}
  {_tile("Rentabilidad bruta típica", _pct(d['rentab'], 1), "antes de gastos e impuestos")}
</div>
{_credito_html(d)}
<div class="resumen"><ul>{"".join(f"<li>{f}</li>" for f in frases(d))}</ul></div>

{por_zona}
{del_mes}

<div class="salto"></div>
<p class="kicker">Informe semanal · {d['ciudad']}</p>
<h1>Oportunidades de la semana</h1>
<p class="periodo">{periodo} · de {_n(d['evaluadas'])} viviendas nuevas, {_n(d['con_comparables'])} tenían
  suficientes similares para compararlas y {_n(d['bajo_umbral'])} estaban al menos un {_pct(UMBRAL)} por debajo. Estas
  son las más claras.</p>
{oportunidades or '<p class="vacio">Esta semana no hubo oportunidades que pasaran los filtros.</p>'}

<div class="salto"></div>
<p class="kicker">Informe semanal · {d['ciudad']}</p>
<h1>Bajadas y captación</h1>
<p class="periodo">{periodo}</p>
<h2>Bajaron de precio esta semana</h2>
<p class="sub">{_n(d['bajadas'])} viviendas bajaron {_pct(op.BAJA_MIN)} o más; estas son las más interesantes.</p>
{f'''<table class="mini"><thead><tr><th>Inmueble</th><th>Zona</th><th class="num">Antes</th><th class="num">Ahora</th>
  <th class="num">Bajó</th><th class="num">Frente a similares</th><th></th></tr></thead><tbody>{bajadas}</tbody></table>'''
  if bajadas else '<p class="vacio">Esta semana no hubo bajadas de precio.</p>'}
<h2>Para captar</h2>
<p class="sub">{_n(d['captar_n'])} viviendas llevan entre 4 y 12 meses publicadas y piden entre un {_pct(CAPTAR_SOBRE[0])} y un
  {_pct(CAPTAR_SOBRE[1])} más que inmuebles similares: propietarios que pueden estar abiertos a un precio realista. Las que
  más tiempo llevan:</p>
{f'''<table class="mini"><thead><tr><th>Zona</th><th>Inmueble</th><th class="num">Precio</th>
  <th class="num">Frente a similares</th><th class="num">Frente al modelo</th><th class="num">Días publicada</th><th>Lo publica</th><th></th></tr></thead><tbody>{captar}</tbody></table>
<p class="nota-c">Lo publica: «Particular» si el portal lo marca así o si el anunciante tiene 1 o 2 avisos y no es una
  empresa (una pista, no un dato del portal); «N anunciantes» si el mismo inmueble está con varios.</p>'''
  if captar else '<p class="vacio">Esta semana no hay candidatas.</p>'}
<p class="nota">Viviendas: apartamentos, apartaestudios y casas. Sin repetidos: un inmueble publicado en varios
  portales o por varias inmobiliarias cuenta una vez, también en los movimientos de la semana (publicadas, retiradas,
  bajadas); retirada: ya no está publicada en ningún portal. Arriendos: sin amoblados ni por días. Similares: mismo tipo y estrato, área ±30% y ±1 habitación,
  en el mismo barrio oficial (o a menos de 1 km, o en la {div}); el descuento compara el precio por m² con el promedio
  de esos similares. Promedio: el valor del medio (la mediana), que no se deja llevar por los avisos con precios extremos. Entorno: OpenStreetMap. Es una estimación del precio de mercado, no un avalúo (Ley 1673 de
  2013): verifica siempre el inmueble y su documentación. Fuente: avisos públicos de portales inmobiliarios{
  f"; {_plural(div)} y barrios: {d['fuente_zonas']}" if d["con_zonas"] and d["fuente_zonas"] else ""}{
  "; crédito de vivienda: Superintendencia Financiera (créditos de compra en pesos, tasa ponderada por monto; la "
  "cuota es una aproximación, no una oferta de crédito: en pesos, fija; en UVR, la primera, que luego sube con la "
  "inflación)" if d.get("credito") else ""}.</p>
</body></html>"""


def pie(d: dict) -> str:
    """El texto que acompaña al PDF en Telegram (formato de src/notificar.py: *negrita*)."""
    return (f"*Informe semanal · {d['ciudad']}*\n{_periodo(d['desde'], d['hasta'])}: "
            f"{_n(d['viviendas'])} viviendas en venta, {_n(d['publicadas'])} nuevas, {_n(d['bajadas'])} bajaron "
            f"de precio, {_n(len(d['sel']))} oportunidades y {_n(d['captar_n'])} candidatas para captar.")


def edicion(d: dict) -> dict:
    """Añade a `d` lo que depende de la semana: la primera de cada mes lleva el mapa y la tabla por comuna y el mes
    anterior (`mensual`, `mes`); las demás, el barrio de la semana. El barrio lo anota el canal cada madrugada en
    contenido/barrios_semana.json: aquí solo se lee (sin anotar), así que es el mismo del carrusel."""
    from src import contenido
    d["mensual"] = contenido.es_semana_del_mes(d["hasta"])
    d["mes"] = contenido.mes(d) if d["mensual"] and d["con_zonas"] else None
    if d["mensual"] and len(d["tabla"]):        # la tabla por comuna, con qué parte del catastro está en venta (E9.2)
        d["tabla"] = contenido.con_catastro(d["tabla"], d["inmuebles"], ["comuna"])
    d["barrio_semana"] = None
    try:
        d["credito"] = contenido.credito_semana(d)
    except Exception as e:              # noqa: BLE001 — sin el crédito, el informe sale igual
        d["credito"] = None
        print(f"⚠️  Sin crédito de vivienda: {type(e).__name__}: {e}")
    if d["con_zonas"]:
        try:
            tb = contenido.por_barrio(d)
            vistos = dict((contenido.leer(contenido.BARRIOS_SEMANA) or {}).get(d["ciudad"], {}))
            elegido = contenido.elegir_barrio(tb, d["hasta"], d["precio_m2"], vistos)
            d["barrio_semana"] = contenido.ficha_barrio(tb, *elegido, d) if elegido else None
        except Exception as e:          # noqa: BLE001 — sin el barrio, el informe sale igual
            print(f"⚠️  Sin barrio de la semana: {type(e).__name__}: {e}")
    return d


def generar(ciudad: str = "Cali", fecha: str | None = None) -> tuple[str, dict]:
    """Escribe el .html y el .pdf; devuelve la ruta del PDF y los datos."""
    from weasyprint import HTML
    d = datos(ciudad, fecha)
    edicion(d)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"{zonas.slug(ciudad)}_{d['hasta']:%Y-%m-%d}"
    texto = informe_html(d)
    base.with_suffix(".html").write_text(texto, encoding="utf-8")
    HTML(string=texto, base_url=str(BASE_DIR)).write_pdf(base.with_suffix(".pdf"))
    print(f"✅ {base.with_suffix('.pdf')}")
    return str(base.with_suffix(".pdf")), d


def enviar(pdf: str, d: dict, a_quien: str) -> None:
    """Por Telegram: 'admin' (TELEGRAM_ADMIN_CHAT_ID) o 'agentes' (TELEGRAM_CHAT_ID, separados por coma)."""
    import os
    from src.notificar import documento_a_todos
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chats = os.environ.get("TELEGRAM_ADMIN_CHAT_ID" if a_quien == "admin" else "TELEGRAM_CHAT_ID", "")
    if not (token and chats):
        raise SystemExit("❌ Falta TELEGRAM_BOT_TOKEN o el chat de destino")
    if fallos := documento_a_todos(pdf, token, chats, pie(d)):
        raise SystemExit("❌ Fallaron algunos envíos del PDF:\n  " + "\n  ".join(fallos))
    print(f"✅ Enviado ({a_quien})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Informe semanal para los agentes (PDF)")
    p.add_argument("--ciudad", default="Cali")
    p.add_argument("--fecha", help="último día de la semana (AAAA-MM-DD); por defecto, el último dato")
    p.add_argument("--enviar", choices=["admin", "agentes"], help="lo envía por Telegram")
    a = p.parse_args()
    pdf, d = generar(a.ciudad, a.fecha)
    if a.enviar:
        try:                                    # en local, las claves vienen de .env
            from src.almacen import _cargar_env
            _cargar_env()
        except ImportError:
            pass
        enviar(pdf, d, a.enviar)
