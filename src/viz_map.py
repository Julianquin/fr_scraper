"""
Mapa interactivo (Folium/Leaflet) de inmuebles por ciudad. 100% descriptivo.

Genera un archivo HTML autónomo que puedes abrir en el navegador. Incluye:
  - Puntos coloreados por precio/m² (verde = barato, rojo = caro).
  - Mapa de calor de densidad de oferta.
  - Agrupación por clústeres para explorar zonas.
  - Popups con precio, área, hab/baños, barrio y enlace al aviso.
  - Panel de estadísticas y control de capas / pantalla completa / minimapa.

Uso:
    python -m src.viz_map --ciudad Cali
    python -m src.viz_map --ciudad "Medellín" --out reports/medellin_map.html
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, FastMarkerCluster, Fullscreen, MiniMap
from branca.colormap import LinearColormap

from src.config import DATA_RAW, BASE_DIR
from src.preprocessing import preprocesar_datos_finca_raiz

# Paleta verde → rojo (barato → caro) para el precio por m²
PALETA = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b", "#fc8d59", "#d73027"]


def _cargar_todo() -> pd.DataFrame:
    """Devuelve el dataset limpio. Prefiere el parquet curado (rápido, y en CI
    no hay CSV crudos); si no existe, lee y limpia todos los CSV de data/raw."""
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


def cargar_ciudad(ciudad: str) -> pd.DataFrame:
    """Deja solo la ciudad pedida con geo válida y precio/m² saneado."""
    df = _cargar_todo()
    df = df[df["Ciudad"].str.contains(ciudad, case=False, na=False)].copy()
    for c in ["Latitud", "Longitud", "Precio", "Area_m2", "Habitaciones", "Baños"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Latitud", "Longitud", "Precio", "Area_m2"])
    df = df[(df["Area_m2"] > 10) & (df["Precio"] > 1e7)]

    # Quitar puntos lejanos (fuera del área urbana) usando la mediana como centro
    lat0, lon0 = df["Latitud"].median(), df["Longitud"].median()
    df = df[(df["Latitud"].sub(lat0).abs() < 0.25) & (df["Longitud"].sub(lon0).abs() < 0.25)]

    # Precio por m² y recorte de outliers absurdos (errores de captura)
    df["precio_m2"] = df["Precio"] / df["Area_m2"]
    lo, hi = df["precio_m2"].quantile([0.01, 0.99])
    df = df[df["precio_m2"].between(lo, hi)]

    if df.empty:
        raise ValueError(f"No quedaron inmuebles con geo válida para '{ciudad}'.")
    return df.reset_index(drop=True)


def _popup_html(row: pd.Series) -> str:
    fmt = lambda x: f"{x:,.0f}".replace(",", ".")
    return f"""
    <div style="font-family:system-ui;font-size:13px;line-height:1.5;min-width:210px">
      <div style="font-weight:600;margin-bottom:4px">{str(row['Título'])[:70]}</div>
      <div>💰 <b>${fmt(row['Precio'])}</b> COP</div>
      <div>📐 {row['Area_m2']:.0f} m² · 🛏 {row['Habitaciones']:.0f} · 🛁 {row['Baños']:.0f}</div>
      <div>🏷️ <b>{row['precio_m2']/1e6:.1f}</b> M/m²</div>
      <div>📍 {row['Barrio']} — {row['Tipo_propiedad']}</div>
      <a href="{row['URL detalle']}" target="_blank">Ver aviso ↗</a>
    </div>"""


def construir_mapa(df: pd.DataFrame, ciudad: str) -> folium.Map:
    centro = [df["Latitud"].median(), df["Longitud"].median()]
    m = folium.Map(location=centro, zoom_start=12, tiles=None, control_scale=True)

    folium.TileLayer("CartoDB positron", name="Claro").add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Oscuro").add_to(m)

    # Escala de color robusta (percentiles 5-95) del precio/m² en millones
    pm = df["precio_m2"] / 1e6
    vmin, vmax = float(pm.quantile(0.05)), float(pm.quantile(0.95))
    cmap = LinearColormap(PALETA, vmin=vmin, vmax=vmax)
    cmap.caption = "Precio por m² (millones COP)"

    # --- Capa 1: puntos por precio/m² ---
    fg_precio = folium.FeatureGroup(name="💰 Precio por m²", show=True)
    for _, r in df.iterrows():
        val = r["precio_m2"] / 1e6
        folium.CircleMarker(
            location=[r["Latitud"], r["Longitud"]],
            radius=4, weight=0.5, color="#333", fill=True,
            fill_color=cmap(val), fill_opacity=0.85,
            popup=folium.Popup(_popup_html(r), max_width=260),
        ).add_to(fg_precio)
    fg_precio.add_to(m)

    # --- Capa 2: mapa de calor de densidad ---
    fg_heat = folium.FeatureGroup(name="🔥 Densidad de oferta", show=False)
    HeatMap(df[["Latitud", "Longitud"]].values.tolist(),
            radius=14, blur=20, min_opacity=0.3).add_to(fg_heat)
    fg_heat.add_to(m)

    # --- Capa 3: clústeres (conteo por zona), liviano ---
    fg_clu = folium.FeatureGroup(name="📍 Agrupados por zona", show=False)
    FastMarkerCluster(df[["Latitud", "Longitud"]].values.tolist()).add_to(fg_clu)
    fg_clu.add_to(m)

    cmap.add_to(m)
    Fullscreen(title="Pantalla completa", title_cancel="Salir").add_to(m)
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    _añadir_panel(m, df, ciudad)
    return m


def _añadir_panel(m: folium.Map, df: pd.DataFrame, ciudad: str) -> None:
    """Panel flotante con título y estadísticas descriptivas."""
    pm = df["precio_m2"] / 1e6
    barrio_top = df["Barrio"].value_counts().idxmax()
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
        <div><div style="font-size:22px;font-weight:700;color:#1a9850">{df['Barrio'].nunique()}</div>
             <div style="color:#888">barrios</div></div>
      </div>
      <div style="font-size:12px;color:#555;margin-top:8px">
        Rango típico: <b>{pm.quantile(.25):.1f}–{pm.quantile(.75):.1f}</b> M/m² ·
        Barrio con más oferta: <b>{barrio_top}</b></div>
    </div>"""
    m.get_root().html.add_child(folium.Element(html))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ciudad", default="Cali", help="Ciudad a mapear (coincide con la columna Ciudad)")
    parser.add_argument("--out", default=None, help="Ruta del HTML de salida")
    args = parser.parse_args()

    df = cargar_ciudad(args.ciudad)
    slug = args.ciudad.lower().replace(" ", "_").replace("í", "i").replace("é", "e")
    out = Path(args.out) if args.out else BASE_DIR / "reports" / f"{slug}_map.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    m = construir_mapa(df, args.ciudad)
    m.save(str(out))
    print(f"✅ {len(df):,} inmuebles mapeados → {out}")
    print(f"   Ábrelo en el navegador:  file://{out.resolve()}")


if __name__ == "__main__":
    main()
