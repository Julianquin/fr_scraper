# streamlit_dashboard.py
# src/train.py
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from src.config import DATA_INT, MODEL_DIR

TARGET = "Precio"

def main():
    df = pd.read_parquet(DATA_INT / "housing_clean.parquet")

    num_cols = ["Area_m2", "Habitaciones", "Baños"]
    cat_cols = ["Tipo_propiedad", "Ciudad", "Departamento",
                "Etiqueta_Proyecto", "Etiqueta_Destacado",
                "Etiqueta_Nuevo", "Etiqueta_Oportunidad"]

    X = df[num_cols + cat_cols]
    y = df[TARGET]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipe = Pipeline([("pre", pre), ("rf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, pipe.predict(X_test))
    print(f"✅ Modelo entrenado. MAE = {mae:,.0f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "model.pkl")
    print(f"🎉 Artefacto guardado en {MODEL_DIR / 'model.pkl'}")

if __name__ == "__main__":
    main()
