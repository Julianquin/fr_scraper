"""
Informe semanal para los agentes: el pulso del mercado de vivienda de una ciudad, sus oportunidades y a quién
captar, en un PDF de cuatro páginas fácil de leer y de reenviar. Solo datos reales: lo que necesita semanas de
historia (tendencias, días hasta venderse, acierto de la alerta) se irá sumando.

  Página 1 · Pulso del mercado
    - viviendas en venta sin repetidos entre FincaRaíz y Metrocuadrado (src/unificar.py); publicadas,
      retiradas y que bajaron de precio en la semana (de FincaRaíz: Metrocuadrado todavía no tiene una
      semana de historia); arriendos de vivienda, canon y rentabilidad bruta típica
    - el resumen en tres frases y el mapa de la ciudad por comuna o localidad, coloreado por precio por m²
  Página 2 · Por comuna o localidad (src/zonas.py, src/geo_ciudades.py; sin polígonos, la ciudad va sin ellas): viviendas en venta, precio mediano por m², días publicados (edad mediana de los
    avisos activos), canon mediano por m² y rentabilidad bruta típica (canon por m² × 12 ÷ precio por m²)
  Página 3 · Oportunidades: las de la alerta (src/oportunidades.py, mismos criterios), cada una con 3 de sus
    comparables, si el mismo inmueble está también en Metrocuadrado y a qué precio, el arriendo estimado y el
    entorno (OpenStreetMap, src/osm_pois.py: MIO, parques, centros comerciales y comercio a 500 m)
  Página 4 · Bajaron de precio y Para captar: viviendas con 4 a 12 meses publicadas que piden entre un 10% y un
    60% más que sus comparables (más que eso suele ser un error del aviso), las que más tiempo llevan

Es una estimación del precio de mercado, no un avalúo (Ley 1673 de 2013). Sin fotos ni datos de los
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

from src import oportunidades as op, rentabilidad, unificar, viz_map, zonas
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

def _retiradas(ciudad: str, desde: pd.Timestamp, hasta: pd.Timestamp) -> int:
    """Viviendas de la ciudad con la baja confirmada en la semana (historial de src/consolidar.py)."""
    from src.consolidar import rutas
    r = rutas("venta")
    todos = pd.read_parquet(r.avisos, columns=["id_inmueble", "ciudad", "tipo", "latitud", "longitud"])
    aqui = (zonas.en_area(ciudad, todos["latitud"], todos["longitud"], todos["ciudad"]) if ciudad in zonas.AREAS
            else todos["ciudad"].map(zonas.canonica) == ciudad)
    viv = todos.loc[aqui & todos["tipo"].isin(op.TIPOS), "id_inmueble"].astype(str)
    dias = pd.date_range(desde, hasta).strftime("%Y-%m-%d")
    partes = [pd.read_parquet(p) for d in dias if (p := r.historial / f"fecha={d}.parquet").exists()]
    if not partes:
        return 0
    h = pd.concat(partes, ignore_index=True)
    return int(((h["evento"] == "baja") & h["id_inmueble"].astype(str).isin(viv)).sum())


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


def _en_metrocuadrado(sel: pd.DataFrame, avisos: pd.DataFrame) -> pd.DataFrame:
    """Para cada oportunidad de FincaRaíz, el mismo inmueble en Metrocuadrado (registro único), si está."""
    grupo = avisos.set_index("clave")["id_unificado"]
    m2 = avisos[avisos["portal"] == "metrocuadrado"].drop_duplicates("id_unificado").set_index("id_unificado")
    g = ("fincaraiz:" + sel["id_inmueble"].astype(str)).map(grupo)
    return pd.DataFrame({"m2_precio": g.map(m2["precio"]).to_numpy(), "m2_url": g.map(m2["url"]).to_numpy()},
                        index=sel.index)


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
    nuevos = venta[venta["first_seen"].between(desde, hasta)]
    ev = op.calificar(venta, nuevos, hist) if len(nuevos) else nuevos
    bj = op.bajadas(venta, hist, desde, hasta)
    arr = rentabilidad.arriendos(ciudad)
    ev = rentabilidad.estimar(ev, arr) if len(ev) else ev
    bj = rentabilidad.estimar(bj, arr) if len(bj) else bj
    sel = op.seleccionar(ev, UMBRAL, TOP) if len(ev) else ev
    sel_bj = op.seleccionar_bajadas(bj, sel.get("id_inmueble", pd.Index([])), TOP_BAJADAS) if len(bj) else bj
    avisos, inm, _ = unificar.unificar(ciudad)
    if len(sel):
        sel = sel.join(_en_metrocuadrado(sel, avisos)).join(_entorno(sel))
        sel["comps"] = [op._comparables(r, venta)[0].assign(_d=lambda c: (c["Area_m2"] - r["Area_m2"]).abs())
                        .nsmallest(3, "_d") for _, r in sel.iterrows()]

    # Para captar: llevan tiempo y piden bastante más que sus comparables
    viejas = venta[venta["dias_pub"].between(*CAPTAR_DIAS)]
    cap = op.calificar(venta, viejas, hist) if len(viejas) else viejas
    if len(cap):
        cap = cap[cap["descuento"].between(-CAPTAR_SOBRE[1], -CAPTAR_SOBRE[0]) & cap["motivo_descarte"].isna()
                  & (cap["confianza"] != "Baja")]

    # Oferta sin repetidos (los dos portales), con su comuna y un precio por m² saneado como el de venta
    con_zonas = zonas.tiene_zonas(ciudad)
    inm = inm.rename(columns={"latitud": "Latitud", "longitud": "Longitud", "barrio": "Barrio"})
    inm = zonas.asignar_zonas(inm, ciudad) if con_zonas else inm.assign(barrio_oficial=None, comuna=None)
    inm["precio_m2"] = inm["precio_mediano"] / inm["m2"].where(inm["m2"] > 10)
    lo, hi = venta["precio_m2"].quantile([0.01, 0.99])
    con_precio = inm[inm["precio_m2"].between(lo, hi) & (inm["precio_mediano"] > 1e7)]
    arr_ok = arr[arr["motivo_exclusion"].isna()]
    todas = _por_comuna(con_precio, arr_ok, venta) if con_zonas else pd.DataFrame()      # el mapa: todas
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
        "publicadas": len(nuevos), "retiradas": _retiradas(ciudad, desde, hasta),
        "bajadas": len(bj), "bajada_mediana": float(bj["baja"].median()) if len(bj) else None,
        "arriendos": len(arr_ok), "canon_mediano": float(arr_ok["Precio"].median()),
        "precio_m2": float(precio_m2), "rentab": float(12 * canon_m2 / precio_m2),
        "tabla": tabla, "todas": todas, "barrios": barrios, "puntos": con_precio[["Latitud", "Longitud", "comuna"]],
        "evaluadas": len(ev), "con_comparables": int(ev["nivel_comps"].notna().sum()) if len(ev) else 0,
        "bajo_umbral": int((ev["descuento"] >= UMBRAL).sum()) if len(ev) else 0,
        "sel": sel, "sel_bj": sel_bj,
        "captar_n": len(cap), "captar": cap.sort_values("dias_pub", ascending=False).head(CAPTAR_FILAS) if len(cap) else cap,
    }


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
        f"En {d['ciudad']} hay <b>{_n(d['viviendas'])} viviendas en venta</b> sin contar repetidos: "
        f"{_n(d['avisos_fr'])} avisos en FincaRaíz y {_n(d['avisos_m2'])} en Metrocuadrado. "
        f"{_n(d['en_los_dos'])} están en los dos portales y {_n(d['solo_m2'])} solo en Metrocuadrado.",
        f"Esta semana se publicaron <b>{_n(d['publicadas'])}</b> viviendas, se confirmó el retiro de "
        f"<b>{_n(d['retiradas'])}</b> y <b>{_n(d['bajadas'])}</b> bajaron de precio"
        + (f" (rebaja mediana del {_pct(d['bajada_mediana'])})." if d["bajada_mediana"] else "."),
    ]
    if len(t):
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
    return f"""
      <tr>
        <td class="c-name"><b>{html.escape(nombre)}</b> <span>{html.escape(str(r['barrios']))}</span></td>
        <td class="num">{_n(r['viviendas'])}</td>
        <td class="num">{_mill(r['precio_m2'])}</td>
        <td class="bar-cell">{barra(r['precio_m2'], pmax, 'b-precio')}</td>
        <td class="num">{r['dias_pub']:.0f}</td>
        <td class="num">{canon}</td>
        <td class="num">{rent}</td>
        <td class="bar-cell narrow">{barra(r['rentab'], rmax, 'b-rentab')}</td>
      </tr>"""


def _oportunidad(r: pd.Series, i: int, d: dict) -> str:
    hasta = d["hasta"]
    dias = (hasta - r["first_seen"]).days
    lineas = [f"<b>{r['n_comps']} similares</b> {'en el barrio' if r['nivel_comps'] == 'barrio' else 'en la zona'}, "
              f"mediana {_mill(r['med_comps'])}/m² · confianza {r['confianza'].lower()}"]
    if pd.notna(r.get("m2_precio")):
        dif = r["m2_precio"] / r["Precio"] - 1
        cuanto = "mismo precio" if abs(dif) < .005 else f"{abs(dif):.0%} {'más' if dif > 0 else 'menos'} que en FincaRaíz"
        lineas.append(f'<b>También en Metrocuadrado:</b> {_mill(r["m2_precio"], 0)} ({cuanto}) · '
                      f'<a href="{html.escape(str(r["m2_url"]))}">ver ↗</a>')
    else:
        lineas.append("No aparece en Metrocuadrado")
    if pd.notna(r.get("canon_est")):
        lineas.append(f"Arriendo estimado {_mill(r['canon_est'])}/mes · rentabilidad bruta {_pct(r['rentab_bruta'], 1)} "
                      f"({r['n_arr']:.0f} similares)")
    if pd.notna(r.get("dist_transporte_masivo_km")):
        lineas.append(f"<b>Entorno:</b> {d['transporte']} {_dist(r['dist_transporte_masivo_km'])} · parque "
                      f"{_dist(r['dist_parque_km'])} · c. comercial {_dist(r['dist_centro_comercial_km'])} · "
                      f"{r['n_comercio_500m']:.0f} comercios a 500 m")
    comps = "".join(f"<tr><td>{html.escape(PREFIJO.sub('', str(c['barrio_oficial'] or c['Barrio'])))}</td>"
                    f"<td class='num'>{c['Area_m2']:.0f} m²</td><td class='num'>{_mill(c['Precio'], 0)}</td>"
                    f"<td class='num'>{_mill(c['precio_m2'])}/m²</td></tr>" for _, c in r["comps"].iterrows())
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
      <div><p class="mini-tit">3 de sus similares, los de área más parecida</p>
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
              f'todos los avisos de los dos portales, por barrio y con su enlace</p>')
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
    <th class="num">Precio mediano<br>por m²</th><th></th><th class="num">Días<br>publicados</th>
    <th class="num">Canon mediano<br>por m²</th><th class="num">Rentabilidad<br>bruta típica</th><th></th></tr></thead>
  <tbody>{filas}</tbody>
</table>
<p class="nota">Días publicados: edad mediana de los avisos activos de la {div} (desde su fecha de publicación).
  Rentabilidad bruta típica: canon mediano por m² × 12 ÷ precio mediano por m², antes de administración, impuestos y
  vacancia; «—»: menos de {MIN_ARRIENDOS} arriendos.</p>"""


def informe_html(d: dict) -> str:
    t, nombres, div = d["tabla"], d["nombres"], d["division"]
    periodo = _periodo(d["desde"], d["hasta"])
    por_zona = _por_zona_html(d, periodo) if len(t) else (
        f'<p class="nota">Todavía no tenemos los polígonos oficiales de {_plural(div)} y barrios de {d["ciudad"]}: '
        f'el informe va sin mapa ni tabla por {div}, y los similares se buscan a menos de 1 km.</p>')
    oportunidades = "".join(_oportunidad(r, i, d) for i, (_, r) in enumerate(d["sel"].iterrows(), 1))
    bajadas = "".join(
        f"<tr><td>{html.escape(op._ficha(r))}</td><td>{html.escape(_barrio(r, nombres))}</td>"
        f"<td class='num'>{_mill(r['precio_antes'], 0)}</td><td class='num'><b>{_mill(r['Precio'], 0)}</b></td>"
        f"<td class='num'>−{_pct(r['baja'])}</td>"
        f"<td class='num'>{html.escape((op._frente_comps(r) or '—').replace('ahora ', '').replace('sus comparables', 'similares'))}</td>"
        f"<td><a href='{html.escape(str(r['URL detalle']))}'>ver ↗</a></td></tr>" for _, r in d["sel_bj"].iterrows())
    captar = "".join(
        f"<tr><td>{html.escape(_barrio(r, nombres))}</td><td>{html.escape(r['Tipo_propiedad'])} · {r['Area_m2']:.0f} m²</td>"
        f"<td class='num'>{_mill(r['Precio'], 0)}</td><td class='num'>{_pct(-r['descuento'])} más</td>"
        f"<td class='num'>{r['dias_pub']:.0f}</td><td><a href='{html.escape(str(r['URL detalle']))}'>ver ↗</a></td></tr>"
        for _, r in d["captar"].iterrows())
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Pulso del mercado · {d['ciudad']}</title>
<style>
  @page {{ size: A4; margin: 12mm 14mm 14mm;
    @bottom-left {{ content: "Informe semanal · {d['ciudad']} · {_fecha(d['hasta'])}"; font: 7pt Inter, sans-serif; color: #898781; }}
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
  .opo {{ border: 1px solid var(--line); border-radius: 6pt; padding: 7pt 10pt 8pt; margin-bottom: 7pt; break-inside: avoid; }}
  .opo-top {{ overflow: hidden; margin-bottom: 4pt; }}   /* sin flex: WeasyPrint pierde los enlaces dentro de flex o grid */
  .opo-top .ver {{ float: right; }}
  .opo-cuerpo {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 12pt; }}
  .badge {{ font-weight: 600; font-size: 7.6pt; border-radius: 99pt; padding: 1.5pt 7pt; background: var(--good-bg); color: var(--good); }}
  .badge.ojo {{ background: #fdf1dc; color: #7a4a00; }}
  .opo .zona {{ color: var(--ink-2); margin: 1pt 0 3pt; }}
  .opo .precio {{ font-size: 9.2pt; margin: 0 0 3pt; }}
  .opo .hecho {{ margin: 0 0 2pt; color: var(--ink-2); font-size: 7.7pt; }} .opo .hecho b {{ color: var(--ink); font-weight: 600; }}
  a {{ color: var(--blue); font-weight: 600; text-decoration: none; }}
  .mini-tit {{ color: var(--ink-2); font-size: 7.4pt; margin: 0 0 2pt; }}
  table.mini td, table.mini th {{ font-size: 7.6pt; padding: 2.2pt 3pt; }}
  .vacio {{ color: var(--ink-2); }}
</style></head><body>

<p class="kicker">Informe semanal · vivienda en venta y arriendo</p>
<h1>Pulso del mercado · {d['ciudad']}</h1>
<p class="periodo">{periodo} · datos de FincaRaíz y Metrocuadrado</p>
<div class="tiles">
  {_tile("Viviendas en venta", _n(d['viviendas']), f"sin repetidos · {_n(d['avisos_fr'] + d['avisos_m2'])} avisos")}
  {_tile("Publicadas esta semana", _n(d['publicadas']), "viviendas nuevas en FincaRaíz")}
  {_tile("Retiradas esta semana", _n(d['retiradas']), "confirmadas tras 3 días sin verse")}
  {_tile("Bajaron de precio", _n(d['bajadas']), f"rebaja mediana del {_pct(d['bajada_mediana'])}" if d['bajada_mediana'] else "")}
</div>
<div class="tiles">
  {_tile("Precio mediano por m²", _mill(d['precio_m2']), "vivienda en venta, toda la ciudad")}
  {_tile("Arriendos de vivienda", _n(d['arriendos']), "sin amoblados ni por días")}
  {_tile("Canon mediano", _mill(d['canon_mediano']), "por mes")}
  {_tile("Rentabilidad bruta típica", _pct(d['rentab'], 1), "antes de gastos e impuestos")}
</div>
<div class="resumen"><ul>{"".join(f"<li>{f}</li>" for f in frases(d))}</ul></div>

{por_zona}

<div class="salto"></div>
<p class="kicker">Informe semanal · {d['ciudad']}</p>
<h1>Oportunidades de la semana</h1>
<p class="periodo">{periodo} · de {_n(d['evaluadas'])} viviendas publicadas, {_n(d['con_comparables'])} tenían
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
  <th class="num">Frente a similares</th><th class="num">Días publicada</th><th></th></tr></thead><tbody>{captar}</tbody></table>'''
  if captar else '<p class="vacio">Esta semana no hay candidatas.</p>'}
<p class="nota">Viviendas: apartamentos, apartaestudios y casas. Sin repetidos: un inmueble publicado en los dos
  portales o por varias inmobiliarias cuenta una vez. Los movimientos de la semana (publicadas, retiradas, bajadas)
  son de FincaRaíz. Arriendos: sin amoblados ni por días. Similares: mismo tipo y estrato, área ±30% y ±1 habitación,
  en el mismo barrio oficial (o a menos de 1 km, o en la {div}); el descuento compara el precio por m² con la mediana
  de esos similares. Entorno: OpenStreetMap. Es una estimación del precio de mercado, no un avalúo (Ley 1673 de
  2013): verifica siempre el inmueble y su documentación. Fuente: avisos públicos de FincaRaíz y Metrocuadrado{
  f"; {_plural(div)} y barrios: {d['fuente_zonas']}" if d["con_zonas"] and d["fuente_zonas"] else ""}.</p>
</body></html>"""


def pie(d: dict) -> str:
    """El texto que acompaña al PDF en Telegram (formato de src/notificar.py: *negrita*)."""
    return (f"*Informe semanal · {d['ciudad']}*\n{_periodo(d['desde'], d['hasta'])}: "
            f"{_n(d['viviendas'])} viviendas en venta, {_n(d['publicadas'])} nuevas, {_n(d['bajadas'])} bajaron "
            f"de precio, {_n(len(d['sel']))} oportunidades y {_n(d['captar_n'])} candidatas para captar.")


def generar(ciudad: str = "Cali", fecha: str | None = None) -> tuple[str, dict]:
    """Escribe el .html y el .pdf; devuelve la ruta del PDF y los datos."""
    from weasyprint import HTML
    d = datos(ciudad, fecha)
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
    from src.notificar import documento
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chats = os.environ.get("TELEGRAM_ADMIN_CHAT_ID" if a_quien == "admin" else "TELEGRAM_CHAT_ID", "")
    if not (token and chats):
        raise SystemExit("❌ Falta TELEGRAM_BOT_TOKEN o el chat de destino")
    for chat in filter(None, map(str.strip, chats.split(","))):
        documento(pdf, token, chat, pie(d))
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
