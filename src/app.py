import streamlit as st, joblib, pandas as pd
from src.config import MODEL_DIR

st.title("Dashboard Predictivo del Precio de Viviendas 🏠")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model.pkl")

model = load_model()

# --- Inputs UI ---
area       = st.sidebar.slider("Área (m²)", 20, 500, 70)
habit      = st.sidebar.slider("Habitaciones", 0, 10, 2)
banos      = st.sidebar.slider("Baños", 0, 10, 1)

tipo       = st.sidebar.text_input("Tipo de Propiedad", "Apartamento")
ciudad     = st.sidebar.text_input("Ciudad", "Bogotá")
depto      = st.sidebar.text_input("Departamento", "Cundinamarca")

etiquetas  = {
    "Proyecto"   : st.sidebar.checkbox("Proyecto"),
    "Destacado"  : st.sidebar.checkbox("Destacado"),
    "Nuevo"      : st.sidebar.checkbox("Nuevo"),
    "Oportunidad": st.sidebar.checkbox("Oportunidad"),
}

input_df = pd.DataFrame([{
    "Area_m2": area, "Habitaciones": habit, "Baños": banos,
    "Tipo_propiedad": tipo, "Ciudad": ciudad, "Departamento": depto,
    **{f"Etiqueta_{k}": str(v) for k, v in etiquetas.items()}
}])

pred = model.predict(input_df)[0]
st.metric("💰 Precio estimado", f"${pred:,.0f} COP")


