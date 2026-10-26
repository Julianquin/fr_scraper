"""
Mapa interactivo (Folium/Leaflet) de inmuebles por ciudad. 100% descriptivo.

Genera un archivo HTML autónomo que puedes abrir en el navegador. Incluye:
  - Zonas oficiales coloreadas por mediana de precio/m² (solo Cali, ver src/zonas.py):
    comunas al alejarse, barrios al acercarse (verde = barato, rojo = caro).
  - Avisos individuales al hacer zoom, con popup (precio, área, hab/baños, enlace).
  - Panel de estadísticas, pantalla completa y minimapa.
  En otras ciudades (sin polígonos) muestra solo los avisos.

Uso:
    python -m src.viz_map --ciudad Cali
    python -m src.viz_map --ciudad "Medellín" --out reports/medellin_map.html
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import folium
from folium.plugins import Fullscreen, MiniMap
from branca.colormap import LinearColormap

from src.config import DATA_RAW, BASE_DIR
from src.preprocessing import preprocesar_datos_finca_raiz
from src import zonas

# Paleta verde → rojo (barato → caro) para el precio por m²
PALETA = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b", "#fc8d59", "#d73027"]
ESRI_CANVAS = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas"
MIN_AVISOS = 5          # menos avisos que esto → zona en gris ("pocos datos")
ZOOM_BARRIOS = 13       # desde este zoom se ven barrios en vez de comunas
ZOOM_AVISOS = 15        # desde este zoom aparecen los avisos individuales


FUENTE = "master"     # "master": scraper actual · "http": barrido por segmentos (src/fuente_http.py)


def _cargar_todo() -> pd.DataFrame:
    """Devuelve el dataset limpio. Prefiere el parquet curado (rápido, y en CI
    no hay CSV crudos); si no existe, lee y limpia todos los CSV de data/raw."""
    if FUENTE == "http":
        from src import fuente_http
        return fuente_http.dataset_activo()
    parquets = [BASE_DIR / "data" / "app" / "housing_clean.parquet",
                BASE_DIR / "data" / "processed" / "housing_clean.parquet"]
    p = next((q for q in parquets if q.exists()), None)
    if p is not None:
        return pd.read_parquet(p)

    files = sorted(Path(DATA_RAW).rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay parquet curado ni CSV en {DATA_RAW}")
    return preprocesar_datos_finca_raiz(pd.concat([pd.read_csv(f) for f in files],
                                                  ignore_index=True))


def con_zonas(ciudad: str) -> bool:
    """Hay polígonos oficiales para esta ciudad (por ahora solo Cali)."""
    return zonas.normalizar(ciudad) == "cali"


def cargar_ciudad(ciudad: str) -> pd.DataFrame:
    """Avisos de la ciudad con precio/m² saneado. Las coordenadas son opcionales:
    sin ellas el aviso no se pinta como punto, pero cuenta en su barrio si se
    puede asignar por nombre."""
    df = _cargar_todo()
    # Nombre exacto: con "contiene", "Cali" también traía "Calima" y "Cali, Yumbo"
    df = df[df["Ciudad"].map(zonas.normalizar) == zonas.normalizar(ciudad)].copy()
    for c in ["Latitud", "Longitud", "Precio", "Area_m2", "Habitaciones", "Baños"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Precio", "Area_m2"])
    df = df[(df["Area_m2"] > 10) & (df["Precio"] > 1e7)]

    # Coordenadas lejanas (fuera del área urbana) se descartan como coordenadas
    lat0, lon0 = df["Latitud"].median(), df["Longitud"].median()
    lejos = (df["Latitud"].sub(lat0).abs() >= 0.25) | (df["Longitud"].sub(lon0).abs() >= 0.25)
    df.loc[lejos, ["Latitud", "Longitud"]] = None

    # Precio por m² y recorte de outliers absurdos (errores de captura)
    df["precio_m2"] = df["Precio"] / df["Area_m2"]
    lo, hi = df["precio_m2"].quantile([0.01, 0.99])
    df = df[df["precio_m2"].between(lo, hi)]

    if con_zonas(ciudad):
        df = zonas.asignar_zonas(df)
        df = df[df["barrio_oficial"].notna() | df["Latitud"].notna()]
    else:
        df = df.dropna(subset=["Latitud", "Longitud"])
    if df.empty:
        raise ValueError(f"No quedaron inmuebles válidos para '{ciudad}'.")
    return df.reset_index(drop=True)


def _popup_html(row: pd.Series) -> str:
    fmt = lambda x: f"{x:,.0f}".replace(",", ".")
    barrio = row.get("barrio_oficial") or row["Barrio"]
    return f"""
    <div style="font-family:system-ui;font-size:13px;line-height:1.5;min-width:210px">
      <div style="font-weight:600;margin-bottom:4px">{str(row['Título'])[:70]}</div>
      <div>💰 <b>${fmt(row['Precio'])}</b> COP</div>
      <div>📐 {row['Area_m2']:.0f} m² · 🛏 {row['Habitaciones']:.0f} · 🛁 {row['Baños']:.0f}</div>
      <div>🏷️ <b>{row['precio_m2']/1e6:.1f}</b> M/m²</div>
      <div>📍 {barrio} — {row['Tipo_propiedad']}</div>
      <a href="{row['URL detalle']}" target="_blank">Ver aviso ↗</a>
    </div>"""


def _capa_zonas(df: pd.DataFrame, nivel: str, col: str, cmap: LinearColormap) -> folium.GeoJson:
    """Polígonos de `nivel` coloreados por mediana de precio/m² de los avisos en `col`."""
    pm = (df["precio_m2"] / 1e6).groupby(df[col])
    stats = pd.DataFrame({"n": pm.size(), "med": pm.median(),
                          "p25": pm.quantile(.25), "p75": pm.quantile(.75)})
    gj = zonas.cargar(nivel)
    clave = "nombre" if nivel == "barrios" else "codigo"
    for f in gj["features"]:
        p = f["properties"]
        s = stats.loc[p[clave]] if p[clave] in stats.index else None
        n = int(s["n"]) if s is not None else 0
        titulo = p["nombre"] if nivel == "comunas" else f"{p['nombre']} · Comuna {p['comuna']}"
        if n >= MIN_AVISOS:
            p["color"] = cmap(s["med"])
            p["info"] = (f"<b>{titulo}</b><br>{n:,} avisos · mediana <b>{s['med']:.1f}</b> M/m²"
                         f"<br>rango típico {s['p25']:.1f}–{s['p75']:.1f} M/m²")
        else:
            p["color"] = "#9e9e9e"
            p["info"] = f"<b>{titulo}</b><br>Pocos datos ({n} avisos)"

    return folium.GeoJson(
        gj, name=nivel, control=False,
        style_function=lambda f: {"fillColor": f["properties"]["color"], "fillOpacity": 0.7,
                                  "color": "#ffffff", "weight": 1, "opacity": 0.8},
        highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=["info"], labels=False, sticky=True),
    )


def construir_mapa(df: pd.DataFrame, ciudad: str) -> folium.Map:
    geo = df.dropna(subset=["Latitud", "Longitud"])
    centro = [geo["Latitud"].median(), geo["Longitud"].median()]
    m = folium.Map(location=centro, zoom_start=12, tiles=None, control_scale=True,
                   prefer_canvas=True)

    # Basemaps de Esri (sin API key; CARTO empezó a exigirla)
    for nombre, capa in [("Claro", "World_Light_Gray_Base"), ("Oscuro", "World_Dark_Gray_Base")]:
        folium.TileLayer(
            tiles=f"{ESRI_CANVAS}/{capa}/MapServer/tile/{{z}}/{{y}}/{{x}}",
            attr="Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
            name=nombre, max_zoom=18,
        ).add_to(m)

    # Escala de color robusta (percentiles 5-95) del precio/m² en millones. Con zonas,
    # sobre las medianas de barrio: con la de avisos sueltos todas las zonas salen verdes.
    pm = df["precio_m2"] / 1e6
    if con_zonas(ciudad):
        por_barrio = pm.groupby(df["barrio_oficial"])
        pm = por_barrio.median()[por_barrio.size() >= MIN_AVISOS]
    vmin, vmax = float(pm.quantile(0.05)), float(pm.quantile(0.95))
    cmap = LinearColormap(PALETA, vmin=vmin, vmax=vmax)
    cmap.caption = "Precio por m² (millones COP)"

    # --- Zonas oficiales (comunas / barrios) ---
    capas_zona = {}
    if con_zonas(ciudad):
        capas_zona["comunas"] = _capa_zonas(df, "comunas", "comuna", cmap).add_to(m)
        capas_zona["barrios"] = _capa_zonas(df, "barrios", "barrio_oficial", cmap).add_to(m)

    # --- Avisos individuales ---
    fg_avisos = folium.FeatureGroup(name="Avisos", control=False)
    for _, r in geo.iterrows():
        folium.CircleMarker(
            location=[r["Latitud"], r["Longitud"]],
            radius=5, weight=0.8, color="#333", fill=True,
            fill_color=cmap(r["precio_m2"] / 1e6), fill_opacity=0.9,
            popup=folium.Popup(_popup_html(r), max_width=260),
        ).add_to(fg_avisos)
    fg_avisos.add_to(m)

    cmap.add_to(m)
    Fullscreen(title="Pantalla completa", title_cancel="Salir").add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    if capas_zona:
        _alternar_por_zoom(m, capas_zona["comunas"], capas_zona["barrios"], fg_avisos)
    _añadir_panel(m, df, ciudad)
    return m


def _alternar_por_zoom(m: folium.Map, comunas, barrios, avisos) -> None:
    """Comunas de lejos → barrios al acercarse → avisos (sobre barrios tenues) de cerca."""
    # Corre en DOMContentLoaded: folium declara el mapa y las capas después de este script
    js = f"""
    document.addEventListener('DOMContentLoaded', function() {{
      var map = {m.get_name()}, comunas = {comunas.get_name()},
          barrios = {barrios.get_name()}, avisos = {avisos.get_name()};
      function poner(capa, visible) {{
        if (visible && !map.hasLayer(capa)) map.addLayer(capa);
        if (!visible && map.hasLayer(capa)) map.removeLayer(capa);
      }}
      function sincronizar() {{
        var z = map.getZoom();
        poner(comunas, z < {ZOOM_BARRIOS});
        poner(barrios, z >= {ZOOM_BARRIOS});
        poner(avisos, z >= {ZOOM_AVISOS});
        barrios.setStyle({{fillOpacity: z >= {ZOOM_AVISOS} ? 0.2 : 0.7}});
        if (map.hasLayer(barrios)) barrios.bringToBack();
      }}
      map.on('zoomend', sincronizar);
      sincronizar();
    }});"""
    m.get_root().script.add_child(folium.Element(js))


def _añadir_panel(m: folium.Map, df: pd.DataFrame, ciudad: str) -> None:
    """Panel flotante con título y estadísticas descriptivas."""
    pm = df["precio_m2"] / 1e6
    col = "barrio_oficial" if "barrio_oficial" in df else "Barrio"
    barrio_top = df[col].value_counts().idxmax()
    guia = (f"""<div style="font-size:12px;color:#555;margin-top:6px">
        🔍 Acércate: <b>comunas → barrios → avisos</b><br>
        <span style="color:#888">Zonas: {zonas.FUENTE}</span></div>"""
            if col == "barrio_oficial" else "")
    html = f"""
    <div style="position:fixed;top:14px;left:60px;z-index:9999;
        background:rgba(255,255,255,.94);padding:14px 18px;border-radius:12px;
        box-shadow:0 4px 18px rgba(0,0,0,.18);font-family:system-ui;max-width:290px">
      <div style="font-size:19px;font-weight:700;color:#1a1a2e">🏙️ {ciudad}</div>
      <div style="font-size:12px;color:#666;margin-bottom:8px">
        Mercado de venta · datos FincaRaíz (descriptivo)</div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;font-size:13px">
        <div><div style="font-size:22px;font-weight:700;color:#2c3e50">{len(df):,}</div>
             <div style="color:#888">avisos</div></div>
        <div><div style="font-size:22px;font-weight:700;color:#d73027">{pm.median():.1f}</div>
             <div style="color:#888">M/m² (mediana)</div></div>
        <div><div style="font-size:22px;font-weight:700;color:#1a9850">{df[col].nunique()}</div>
             <div style="color:#888">barrios</div></div>
      </div>
      <div style="font-size:12px;color:#555;margin-top:8px">
        Rango típico: <b>{pm.quantile(.25):.1f}–{pm.quantile(.75):.1f}</b> M/m² ·
        Barrio con más oferta: <b>{barrio_top}</b></div>{guia}
    </div>"""
    m.get_root().html.add_child(folium.Element(html))


def estadisticas(df: pd.DataFrame, ciudad: str) -> dict:
    """Cifras que muestra docs/index.html (se escriben como JSON junto al mapa)."""
    pm = df["precio_m2"] / 1e6
    col = "barrio_oficial" if "barrio_oficial" in df else "Barrio"
    return {
        "ciudad": ciudad,
        "avisos": int(len(df)),
        "mediana_m2": round(float(pm.median()), 2),
        "p25_m2": round(float(pm.quantile(.25)), 2),
        "p75_m2": round(float(pm.quantile(.75)), 2),
        "barrios": int(df[col].nunique()),
        "actualizado": pd.Timestamp.now(tz="America/Bogota").strftime("%Y-%m-%d"),
    }


def main() -> None:
    global FUENTE
    parser = argparse.ArgumentParser()
    parser.add_argument("--ciudad", default="Cali", help="Ciudad a mapear (coincide con la columna Ciudad)")
    parser.add_argument("--out", default=None, help="Ruta del HTML de salida")
    parser.add_argument("--fuente", choices=["master", "http"], default="master",
                        help="datos del scraper actual (master) o del barrido por segmentos (http)")
    args = parser.parse_args()
    FUENTE = args.fuente

    df = cargar_ciudad(args.ciudad)
    slug = args.ciudad.lower().replace(" ", "_").replace("í", "i").replace("é", "e")
    out = Path(args.out) if args.out else BASE_DIR / "reports" / f"{slug}_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    m = construir_mapa(df, args.ciudad)
    m.save(str(out))
    stats = out.with_suffix(".json")
    stats.write_text(json.dumps(estadisticas(df, args.ciudad), ensure_ascii=False, indent=2),
                     encoding="utf-8")
    print(f"✅ {len(df):,} inmuebles mapeados → {out} (+ {stats.name})")
    print(f"   Ábrelo en el navegador:  file://{out.resolve()}")


if __name__ == "__main__":
    main()
