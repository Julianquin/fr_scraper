"""
Arriendo estimado y rentabilidad bruta de avisos de venta (bloque E3).

Para cada aviso de venta busca ARRIENDOS comparables con el mismo criterio que la alerta
(src/oportunidades.py): mismo tipo y estrato, área ±30%, ±1 habitación; en su barrio oficial,
a menos de 1 km o en su comuna.

    canon estimado     = mediana del canon/m² de los comparables × área del aviso
    rentabilidad bruta = canon estimado × 12 ÷ precio de venta

Arriendos que no sirven de comparable (otro mercado, con canon más alto):
  - amoblados: amenidad "Amoblado" o "amoblado/amueblado" en la descripción
  - por días, temporada o vacacional (en la descripción)

Son cánones PEDIDOS, no pagados, y la rentabilidad es bruta: no descuenta administración,
impuestos, vacancia ni comisiones.

Rentabilidad típica (con_tipica, para el informe de captación): cuando no hay arriendos similares, el canon
sale de la rentabilidad bruta típica de las viviendas en venta del mismo tipo cerca (a menos de 2 km; si no,
del mismo estrato; si no, de la ciudad) × el precio de venta que estima el modelo (src/modelo_precio.py).
Con arriendos similares, se promedian los dos cánones. Medido el 7-nov con arriendos reales a los que se les
estimó el canon sin mirarlo (Cali y alrededores, 1.951; Bogotá y la Sabana, 10.769), error mediano:
  - con similares:  similares 15% · 14%  →  promedio con la típica 13% · 12%
  - sin similares (el 32% y el 10% de los arriendos): típica 23% · 21% (antes no había canon)
Rangos (la mitad central del canon real ÷ el estimado en esa prueba): típica 0,79–1,17; promedio 0,86–1,10.

Uso:
    python -m src.rentabilidad --ciudad Cali    # → reports/rentabilidad/cali_<fecha>.csv + resumen
"""
from __future__ import annotations
import argparse
import re

import numpy as np
import pandas as pd

from src import fuente_http
from src.config import BASE_DIR
from src.consolidar import rutas
from src.oportunidades import TIPOS, _comparables
from src.viz_map import limpiar_ciudad

OUT_DIR = BASE_DIR / "reports" / "rentabilidad"
MIN_COMPS = 5           # arriendos comparables mínimos (hay ~6 veces menos arriendos que ventas)
CANON_MIN = 300_000     # menos que esto es un error o una habitación
_AMOBLADO = re.compile(r"\bamoblad|\bamueblad", re.I)
_TEMPORAL = re.compile(r"por d[ií]as|por noche|temporada|vacacion|airbnb|\bturist", re.I)


def disponible() -> bool:
    """¿Hay estado de arriendos en disco? (src/consolidar.py --raiz arriendo)"""
    return rutas("arriendo").avisos.exists()


def motivo_exclusion(a: pd.DataFrame) -> pd.Series:
    """Por qué un arriendo no sirve de comparable ('amoblado' | 'temporal'), o None."""
    texto = a["Descripcion"].fillna("")
    amoblado = a["Amenidades"].fillna("").str.contains("Amoblado") | texto.str.contains(_AMOBLADO)
    temporal = texto.str.contains(_TEMPORAL)
    return pd.Series(np.select([temporal, amoblado], ["temporal", "amoblado"], default=None),
                     index=a.index)


def arriendos(ciudad: str) -> pd.DataFrame:
    """Arriendos de vivienda activos de la ciudad, limpios, con zona oficial y `motivo_exclusion`.
    `precio_m2` es el canon por m²."""
    a = limpiar_ciudad(fuente_http.dataset_activo("arriendo"), ciudad, precio_min=CANON_MIN)
    a = a[a["Tipo_propiedad"].isin(TIPOS)].copy()
    for c in ["Estrato", "Habitaciones"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a["motivo_exclusion"] = motivo_exclusion(a)
    return a.reset_index(drop=True)


def estimar(ventas: pd.DataFrame, arr: pd.DataFrame, minimo: int = MIN_COMPS) -> pd.DataFrame:
    """Agrega a cada aviso de venta su canon estimado y su rentabilidad bruta (con rango p25–p75)."""
    pool = arr[arr["motivo_exclusion"].isna()]
    filas = []
    for _, c in ventas.iterrows():
        comps, nivel = _comparables(c, pool, minimo)
        f = {"nivel_arr": nivel, "n_arr": len(comps) if nivel else 0}
        if nivel:
            m2 = comps["precio_m2"]
            f.update(canon_m2=m2.median(), canon_est=m2.median() * c["Area_m2"],
                     rentab_bruta=m2.median() * c["Area_m2"] * 12 / c["Precio"],
                     rentab_p25=m2.quantile(.25) * c["Area_m2"] * 12 / c["Precio"],
                     rentab_p75=m2.quantile(.75) * c["Area_m2"] * 12 / c["Precio"])
        filas.append(f)
    cols = ["nivel_arr", "n_arr", "canon_m2", "canon_est", "rentab_bruta", "rentab_p25", "rentab_p75",
            "confianza_arr"]
    est = pd.DataFrame(filas, index=ventas.index).reindex(columns=cols)
    # Misma escala que los comparables de venta: barrio con 10 o más → alta; comuna → baja
    est["confianza_arr"] = np.select(
        [(est["nivel_arr"] == "barrio") & (est["n_arr"] >= 10), est["nivel_arr"] == "comuna",
         est["nivel_arr"].notna()], ["Alta", "Baja", "Media"], default=None)
    return ventas.drop(columns=cols, errors="ignore").join(est)


MIN_TIPICA = 10             # ventas con rentabilidad estimada para dar una típica
RADIO_TIPICA_KM = 2
MAX_VENTAS_TIPICA = 300     # las más cercanas (o una muestra): basta para la mediana y es rápido
RANGO_TIPICA = (0.79, 1.17)  # mitad central del canon real ÷ estimado sin similares (prueba del 7-nov)
RANGO_MEZCLA = (0.86, 1.10)  # ídem, con similares, promediando los dos


def tipica(c: pd.Series, ventas: pd.DataFrame, arr: pd.DataFrame) -> tuple[float, str, int, str] | None:
    """(rentabilidad bruta típica, dónde, cuántas ventas, nota) de las viviendas en venta como `c` (mismo tipo): la
    mediana de la rentabilidad por arriendos similares de las ventas a menos de RADIO_TIPICA_KM; si no llegan a
    MIN_TIPICA, las de su estrato; si no, las de la zona. Donde hay tan pocos arriendos que ninguna venta tiene
    similares (Girardot: 21), el canon/m² mediano de los arriendos del tipo ÷ el precio/m² mediano de las ventas
    (prueba del 7-nov: 23–26% de error, frente a 21–23%). None si ni eso."""
    V = ventas[(ventas["Tipo_propiedad"] == c["Tipo_propiedad"]) & (ventas["id_inmueble"] != c.get("id_inmueble"))]
    V = V[V["Precio"] > 0]
    niveles = []
    if pd.notna(c.get("Latitud")):
        km = np.hypot((V["Latitud"] - c["Latitud"]) * 110.54,
                      (V["Longitud"] - c["Longitud"]) * 111.32 * np.cos(np.radians(c["Latitud"])))
        niveles.append((V.assign(_km=km)[km <= RADIO_TIPICA_KM].nsmallest(MAX_VENTAS_TIPICA, "_km"),
                        f"a menos de {RADIO_TIPICA_KM} km"))
    if pd.notna(c.get("Estrato")):
        e = V[V["Estrato"] == c["Estrato"]]
        niveles.append((e.sample(min(len(e), MAX_VENTAS_TIPICA), random_state=0), f"de estrato {c['Estrato']:.0f}"))
    niveles.append((V.sample(min(len(V), MAX_VENTAS_TIPICA), random_state=0), "de la zona"))
    for grupo, donde in niveles:
        if len(grupo) < MIN_TIPICA:
            continue
        r = estimar(grupo, arr)["rentab_bruta"]
        r = r[r.between(0.01, 0.3)]
        if len(r) >= MIN_TIPICA:
            return float(r.median()), donde, len(r), ""
    R = arr[arr["motivo_exclusion"].isna() & (arr["Tipo_propiedad"] == c["Tipo_propiedad"])]
    if len(R) >= MIN_COMPS and len(V) >= MIN_TIPICA:
        rentab = float(R["precio_m2"].median() * 12 / V["precio_m2"].median())
        if 0.01 <= rentab <= 0.3:
            return rentab, "de la zona", len(V), f"; hay solo {len(R)} arriendos parecidos en la zona"
    return None


def con_tipica(r: pd.Series, venta_modelo: float, ventas: pd.DataFrame, arr: pd.DataFrame) -> pd.Series:
    """Completa el arriendo de `r` (una fila de estimar(), con su `Precio`) con la rentabilidad típica × el precio
    de venta que estima el modelo (`venta_modelo`): sin arriendos similares el canon sale de ahí (`metodo_arr`
    'tipica', confianza baja); con ellos se promedian los dos cánones ('mezcla'). Rango calibrado (RANGO_*).
    Sin típica, queda como estaba ('similares' o vacío)."""
    r = r.copy()
    r["metodo_arr"] = "similares" if pd.notna(r.get("nivel_arr")) else None
    t = tipica(r, ventas, arr) if pd.notna(venta_modelo) else None
    if t is None:
        return r
    r["rentab_tipica"], r["donde_tipica"], r["n_tipica"], r["nota_tipica"] = t
    canon_t = venta_modelo * t[0] / 12
    if pd.notna(r.get("canon_est")):
        canon, rango, r["metodo_arr"] = np.sqrt(r["canon_est"] * canon_t), RANGO_MEZCLA, "mezcla"
    else:
        canon, rango, r["metodo_arr"], r["confianza_arr"] = canon_t, RANGO_TIPICA, "tipica", "Baja"
    r["canon_est"] = canon
    r["rentab_bruta"] = canon * 12 / r["Precio"]
    r["rentab_p25"], r["rentab_p75"] = (canon * f * 12 / r["Precio"] for f in rango)
    r["canon_p25"], r["canon_p75"] = (canon * f for f in rango)
    return r


def texto(r: pd.Series) -> str | None:
    """'Arriendo estimado $2,3 M/mes → rentabilidad bruta 7,2% (rango 6,1–8,4%) · 12 arriendos
    similares en el barrio · confianza alta'; None si no hay arriendos comparables."""
    if pd.isna(r.get("nivel_arr")):
        return None
    donde = {"barrio": "en el barrio", "1 km": "a menos de 1 km",
             "comuna": f"en la comuna {r['comuna']}"}[r["nivel_arr"]]
    pct = lambda x: f"{x * 100:.1f}".replace(".", ",")
    canon = f"{r['canon_est'] / 1e6:.1f}".replace(".", ",")
    return (f"Arriendo estimado ${canon} M/mes → rentabilidad bruta {pct(r['rentab_bruta'])}% "
            f"(rango {pct(r['rentab_p25'])}–{pct(r['rentab_p75'])}%) · {r['n_arr']:.0f} arriendos similares "
            f"{donde} · confianza {r['confianza_arr'].lower()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Arriendo estimado y rentabilidad bruta de los avisos de venta")
    p.add_argument("--ciudad", default="Cali")
    p.add_argument("--minimo", type=int, default=MIN_COMPS, help="arriendos comparables mínimos")
    args = p.parse_args()

    arr = arriendos(args.ciudad)
    ventas = limpiar_ciudad(fuente_http.dataset_activo("venta"), args.ciudad)
    ventas = ventas[ventas["Tipo_propiedad"].isin(TIPOS)].copy()
    for c in ["Estrato", "Habitaciones"]:
        ventas[c] = pd.to_numeric(ventas[c], errors="coerce")
    est = estimar(ventas, arr, args.minimo)

    excl = arr["motivo_exclusion"].value_counts().to_dict()
    con = est[est["nivel_arr"].notna()]
    print(f"{args.ciudad}: {len(arr):,} arriendos de vivienda ({len(arr) - sum(excl.values()):,} usables; "
          f"excluidos {excl}) · {len(est):,} avisos de venta")
    print(f"Con arriendo estimado: {len(con):,} ({len(con) / max(len(est), 1):.0%}) · por nivel: "
          f"{est['nivel_arr'].value_counts(dropna=False).to_dict()} · confianza: "
          f"{con['confianza_arr'].value_counts().to_dict()}")
    print("Cobertura por tipo:", est.groupby("Tipo_propiedad")["nivel_arr"].apply(lambda s: f"{s.notna().mean():.0%}").to_dict())
    print(f"Rentabilidad bruta mediana: {con['rentab_bruta'].median():.1%} "
          f"(p25 {con['rentab_bruta'].quantile(.25):.1%} · p75 {con['rentab_bruta'].quantile(.75):.1%})")
    print("\nPor tipo y estrato (mediana · avisos):")
    print(con.groupby(["Tipo_propiedad", "Estrato"])["rentab_bruta"].agg(["median", "size"])
             .query("size >= 20").round(3).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fecha = pd.Timestamp.now().strftime("%Y-%m-%d")
    cols = ["id_inmueble", "Tipo_propiedad", "barrio_oficial", "comuna", "Estrato", "Area_m2", "Habitaciones",
            "Precio", "nivel_arr", "n_arr", "confianza_arr", "canon_m2", "canon_est", "rentab_bruta",
            "rentab_p25", "rentab_p75", "URL detalle"]
    destino = OUT_DIR / f"{args.ciudad.lower()}_{fecha}.csv"
    est[[c for c in cols if c in est]].to_csv(destino, index=False)
    print(f"\n✅ {destino.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
