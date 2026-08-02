"""
Dashboard comercial de inteligencia inmobiliaria (100% descriptivo).

Dos vistas pensadas para el equipo comercial de una inmobiliaria:
  1. 📍 Precio por zona  → dónde está lo caro/barato por m² en la ciudad.
  2. ⚖️ Comparador       → "esta propiedad vs. el mercado de su barrio".

Lee un parquet ya curado (data/app/housing_clean.parquet). Pensado para
desplegar en Streamlit Community Cloud.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Inteligencia Inmobiliaria", page_icon="🏙️", layout="wide")

ESCALA = "RdYlGn_r"  # verde = barato · rojo = caro
DATASETS = ["data/app/housing_clean.parquet", "data/processed/housing_clean.parquet"]


@st.cache_data(show_spinner="Cargando datos de mercado…")
def cargar() -> pd.DataFrame:
    ruta = next((p for p in DATASETS if Path(p).exists()), None)
    if ruta is None:
        st.error("No se encontró el dataset (data/app/housing_clean.parquet).")
        st.stop()
    df = pd.read_parquet(ruta)
    for c in ["Latitud", "Longitud", "Precio", "Area_m2", "Habitaciones", "Baños"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["Precio", "Area_m2"])
    df = df[(df["Area_m2"] > 10) & (df["Precio"] > 1e7)]
    df["precio_m2"] = df["Precio"] / df["Area_m2"]
    return df


def formato_cop(x: float) -> str:
    return f"${x:,.0f}".replace(",", ".")


def recortar_outliers(d: pd.DataFrame) -> pd.DataFrame:
    lo, hi = d["precio_m2"].quantile([0.01, 0.99])
    return d[d["precio_m2"].between(lo, hi)]


# ─────────────────────────── carga + filtros globales ───────────────────────────
df = cargar()

st.sidebar.title("🏙️ Inteligencia Inmobiliaria")
st.sidebar.caption("Demo comercial · datos de mercado · 100% descriptivo")
ciudades = df["Ciudad"].value_counts()
ciudades = ciudades[ciudades >= 20].index.tolist()
ciudad = st.sidebar.selectbox("Ciudad", sorted(ciudades))

d = recortar_outliers(df[df["Ciudad"] == ciudad].copy())
tipos = ["(todos)"] + sorted(d["Tipo_propiedad"].dropna().unique())
tipo_sel = st.sidebar.selectbox("Tipo de propiedad", tipos)
if tipo_sel != "(todos)":
    d = d[d["Tipo_propiedad"] == tipo_sel]

st.sidebar.metric("Avisos en la vista", f"{len(d):,}".replace(",", "."))
st.sidebar.caption("Precio/m² en millones de COP. Se recortan outliers (1%–99%).")

st.title(f"Mercado de vivienda — {ciudad}")

tab_zona, tab_comp = st.tabs(["📍 Precio por zona", "⚖️ Comparador"])

# ─────────────────────────────── VISTA 1: por zona ──────────────────────────────
with tab_zona:
    pm = d["precio_m2"] / 1e6
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avisos", f"{len(d):,}".replace(",", "."))
    c2.metric("Mediana precio/m²", f"{pm.median():.1f} M")
    c3.metric("Precio mediano", formato_cop(d["Precio"].median()))
    c4.metric("Barrios", f"{d['Barrio'].nunique()}")

    geo = d.dropna(subset=["Latitud", "Longitud"])
    geo = geo[(geo["Latitud"].sub(geo["Latitud"].median()).abs() < 0.25) &
              (geo["Longitud"].sub(geo["Longitud"].median()).abs() < 0.25)]
    if len(geo):
        geo = geo.assign(**{"Precio/m² (M)": (geo["precio_m2"] / 1e6).round(2)})
        fig = px.scatter_map(
            geo, lat="Latitud", lon="Longitud",
            color="Precio/m² (M)", color_continuous_scale=ESCALA,
            range_color=[pm.quantile(0.05), pm.quantile(0.95)],
            hover_name="Barrio",
            hover_data={"Latitud": False, "Longitud": False,
                        "Tipo_propiedad": True, "Area_m2": ":.0f", "Precio": ":,.0f"},
            zoom=11, height=520, map_style="open-street-map",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_colorbar_title="M/m²")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Precio por m² según barrio")
    por_barrio = (d.groupby("Barrio")
                    .agg(avisos=("Precio", "size"),
                         precio_m2_M=("precio_m2", lambda s: s.median() / 1e6),
                         precio_mediano=("Precio", "median"),
                         area_mediana=("Area_m2", "median"))
                    .reset_index())
    por_barrio = por_barrio[por_barrio["avisos"] >= 5].sort_values("precio_m2_M", ascending=False)

    colg, colt = st.columns([3, 2])
    with colg:
        top = por_barrio.head(20)
        fig2 = px.bar(top.sort_values("precio_m2_M"), x="precio_m2_M", y="Barrio",
                      orientation="h", color="precio_m2_M", color_continuous_scale=ESCALA,
                      labels={"precio_m2_M": "Precio/m² (M COP)", "Barrio": ""}, height=560)
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig2, width='stretch')
    with colt:
        st.caption("Barrios con ≥5 avisos, ordenados por precio/m².")
        st.dataframe(
            por_barrio.assign(
                **{"Precio/m² (M)": por_barrio["precio_m2_M"].round(2),
                   "Precio mediano": por_barrio["precio_mediano"].map(formato_cop),
                   "Área med. (m²)": por_barrio["area_mediana"].round(0)}
            )[["Barrio", "avisos", "Precio/m² (M)", "Precio mediano", "Área med. (m²)"]],
            hide_index=True, width='stretch', height=520,
        )

# ─────────────────────────────── VISTA 2: comparador ────────────────────────────
with tab_comp:
    st.caption("Compara una propiedad contra el mercado de su barrio. Ideal para "
               "fijar precio de captación o justificar una oferta.")

    conteo_barrio = d["Barrio"].value_counts()
    barrios_ok = sorted(conteo_barrio[conteo_barrio >= 5].index)
    if not barrios_ok:
        st.info("No hay barrios con suficientes avisos en esta ciudad/tipo.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    barrio = col1.selectbox("Barrio", barrios_ok)
    area = col2.number_input("Área (m²)", min_value=15, max_value=2000, value=80, step=5)
    precio = col3.number_input("Precio de la propiedad (COP)", min_value=0,
                               value=350_000_000, step=10_000_000, format="%d")

    zona = d[d["Barrio"] == barrio]
    zpm = zona["precio_m2"] / 1e6
    mediana = zpm.median()

    k1, k2, k3 = st.columns(3)
    k1.metric(f"Mediana en {barrio}", f"{mediana:.1f} M/m²", help=f"{len(zona)} avisos")
    k2.metric("Rango típico (25–75%)", f"{zpm.quantile(.25):.1f}–{zpm.quantile(.75):.1f} M/m²")
    est_med = mediana * 1e6 * area
    k3.metric("Precio sugerido (mediana × área)", formato_cop(est_med))

    if precio > 0 and area > 0:
        mi_pm2 = (precio / area) / 1e6
        pct_vs_med = (mi_pm2 / mediana - 1) * 100
        percentil = (zpm < mi_pm2).mean() * 100

        st.markdown("#### Resultado")
        r1, r2 = st.columns(2)
        r1.metric("Tu precio/m²", f"{mi_pm2:.1f} M",
                  delta=f"{pct_vs_med:+.0f}% vs. mediana", delta_color="inverse")
        r2.metric("Percentil en su barrio", f"{percentil:.0f}%")

        if pct_vs_med > 12:
            st.warning(f"🔺 Está **{pct_vs_med:.0f}% por encima** de la mediana de {barrio}. "
                       "Puede tardar en venderse o tener margen de negociación.")
        elif pct_vs_med < -12:
            st.success(f"🟢 Está **{abs(pct_vs_med):.0f}% por debajo** de la mediana de {barrio}. "
                       "Precio competitivo / posible oportunidad.")
        else:
            st.info(f"✅ Está **en línea** con el mercado de {barrio} ({pct_vs_med:+.0f}%).")

        fig3 = px.histogram(zona.assign(**{"Precio/m² (M)": zpm}), x="Precio/m² (M)",
                            nbins=30, height=320,
                            title=f"Distribución de precio/m² en {barrio}")
        fig3.add_vline(x=mi_pm2, line_color="#d62728", line_width=3,
                       annotation_text="Tu propiedad", annotation_position="top")
        fig3.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig3, width='stretch')

    st.markdown("#### Comparables en el mismo barrio")
    comp = zona.assign(dif=(zona["Area_m2"] - area).abs()).sort_values("dif").head(8)
    comp = comp.assign(**{
        "Precio": comp["Precio"].map(formato_cop),
        "m²": comp["Area_m2"].round(0),
        "Precio/m² (M)": (comp["precio_m2"] / 1e6).round(1),
        "Hab": comp["Habitaciones"].round(0), "Baños": comp["Baños"].round(0),
    })
    cols = ["Título", "Precio", "m²", "Hab", "Baños", "Precio/m² (M)", "URL detalle"]
    st.dataframe(comp[[c for c in cols if c in comp.columns]],
                 hide_index=True, width='stretch',
                 column_config={"URL detalle": st.column_config.LinkColumn("Aviso", display_text="Ver ↗")})

st.caption("Demostración con fines ilustrativos · datos de fuentes públicas · sin modelado predictivo.")
