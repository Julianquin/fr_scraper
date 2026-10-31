"""
Mapa interactivo por ciudad (E6): una página ligera por ciudad o área metropolitana, para GitHub Pages.

  docs/mapa/<ciudad>.html                   la página (Leaflet y su agrupación de puntos, de cdnjs)
  docs/mapa/datos/<ciudad>.json             los puntos: los avisos activos de FincaRaíz (todos los tipos) y de
                                            Metrocuadrado (apartamentos y casas: lo único que barre). Un inmueble
                                            en los dos portales es un solo punto con el enlace a cada uno
                                            (registro único, src/unificar.py)
  docs/mapa/datos/<ciudad>_{comunas,barrios}.geojson
                                            las zonas, coloreadas por el precio mediano por m² de la vivienda
  docs/mapa/datos/<ciudad>_vecinos.geojson  en una ciudad, los municipios vecinos donde caen avisos que dicen
                                            ser de ella (en gris, con el enlace al mapa de su área)
  docs/mapa/index.html                      el índice de ciudades

Ligero a propósito: los puntos van en un JSON compacto que la página carga al abrir (el mapa de Cali hecho con
folium pesa 18 MB por los popups de cada punto). Sin datos de anunciantes: solo lo que muestra el aviso público
y su enlace. El informe semanal (src/informe_semanal.py) enlaza a la página de su ciudad.

Uso:
    python -m src.mapa_ciudad --ciudad Bogotá
    python -m src.mapa_ciudad --todas            # las ciudades y áreas de src/zonas.py, y el índice
"""
from __future__ import annotations
import argparse
import html
import json

import numpy as np
import pandas as pd

from src import unificar, zonas
from src.config import BASE_DIR

SALIDA = BASE_DIR / "docs" / "mapa"
URL = "https://julianquin.github.io/fr_scraper/mapa"
VIVIENDA = {"Apartamento", "Apartaestudio", "Casa"}
GRUPOS = {                                  # el filtro de la página: tipo → grupo
    "Apartamento": "Vivienda", "Apartaestudio": "Vivienda", "Casa": "Vivienda",
    "Local": "Comercial", "Oficina": "Comercial", "Consultorio": "Comercial", "Bodega": "Comercial",
    "Edificio": "Comercial", "Parqueadero": "Comercial",
    "Lote": "Lotes y campo", "Finca": "Lotes y campo", "Casa Campestre": "Lotes y campo", "Casa Lote": "Lotes y campo",
    "Cabaña": "Lotes y campo",
}
MIN_ZONA = 5                                # una zona lleva color con al menos estas viviendas
MIN_VECINO = 3                              # un municipio vecino se dibuja si caen en él al menos estos avisos
                                            # (uno suelto suele ser una coordenada mal puesta)
RAMPA = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]
DOMINIOS = {"fr": "https://www.fincaraiz.com.co", "m2": "https://www.metrocuadrado.com"}


def _ruta(url, dominio: str) -> str:
    u = str(url) if pd.notna(url) else ""
    return u[len(dominio):] if u.startswith(dominio) else u


def puntos(ciudad: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(puntos del mapa, inmuebles de vivienda sin repetidos para colorear las zonas)."""
    from src.consolidar import rutas
    avisos, inm, _ = unificar.unificar(ciudad)
    fr = avisos[avisos["portal"] == "fincaraiz"].sort_values("precio").drop_duplicates("id_unificado")
    m2 = avisos[avisos["portal"] == "metrocuadrado"].sort_values("precio").drop_duplicates("id_unificado")
    viv = inm.set_index("id_unificado")
    viv = viv.assign(pf=fr.set_index("id_unificado")["precio"], uf=fr.set_index("id_unificado")["url"],
                     pm=m2.set_index("id_unificado")["precio"], um=m2.set_index("id_unificado")["url"]).reset_index()

    # Los demás tipos: solo FincaRaíz (Metrocuadrado solo barre vivienda)
    cols = ["tipo", "precio", "m2", "habitaciones", "latitud", "longitud", "url", "ciudad", "estado_aviso"]
    o = pd.read_parquet(rutas("venta").avisos, columns=cols)
    o = o[(o["estado_aviso"] == "activo") & ~o["tipo"].isin(VIVIENDA)]
    o = o[zonas.en_area(ciudad, o["latitud"], o["longitud"], o["ciudad"])] if ciudad in zonas.AREAS \
        else o[o["ciudad"].map(zonas.canonica) == ciudad]
    otros = pd.DataFrame({"tipo": o["tipo"], "latitud": o["latitud"], "longitud": o["longitud"], "m2": o["m2"],
                          "habitaciones": o["habitaciones"], "pf": o["precio"], "uf": o["url"]})
    p = pd.concat([viv[["tipo", "latitud", "longitud", "m2", "habitaciones", "pf", "uf", "pm", "um"]], otros],
                  ignore_index=True)
    for c in ("latitud", "longitud", "m2", "habitaciones", "pf", "pm"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=["latitud", "longitud", "tipo"])
    p = p[(p["pf"].fillna(0) > 1e7) | (p["pm"].fillna(0) > 1e7)]
    return p, inm


def _zonas(ciudad: str, inm: pd.DataFrame) -> dict[str, dict]:
    """Los GeoJSON de comunas y barrios con n (viviendas), m (mediana en M/m²) y c (clase de color, -1 sin datos)."""
    inm = zonas.asignar_zonas(inm.rename(columns={"latitud": "Latitud", "longitud": "Longitud", "barrio": "Barrio"}), ciudad)
    pm2 = inm["precio_mediano"] / inm["m2"].where(inm["m2"] > 10)
    lo, hi = pm2.quantile([0.01, 0.99])
    inm = inm.assign(pm2=pm2)[pm2.between(lo, hi)]
    nombres = zonas.nombres(ciudad)
    salida = {}
    for nivel, clave in (("comunas", ["comuna"]), ("barrios", ["comuna", "barrio_oficial"])):
        if not zonas.archivo(nivel, ciudad).exists():
            continue
        g = inm.dropna(subset=clave).groupby(clave)["pm2"].agg(["size", "median"])
        con = g[g["size"] >= MIN_ZONA]
        cortes = con["median"].quantile([.2, .4, .6, .8]).to_numpy() if len(con) >= 5 else np.array([])
        gj = zonas.cargar(nivel, ciudad)
        for f in gj["features"]:
            p = f["properties"]
            k = p["codigo"] if nivel == "comunas" else (p.get("comuna"), p["nombre"])
            fila = g.loc[k] if k in g.index else None
            n = int(fila["size"]) if fila is not None else 0
            m = float(fila["median"]) if fila is not None else None
            f["properties"] = {"nombre": nombres.get(p["codigo"], p["nombre"]) if nivel == "comunas" else p["nombre"],
                               "zona": nombres.get(p.get("comuna"), "") if nivel == "barrios" else "",
                               "n": n, "m": round(m / 1e6, 2) if m and n >= MIN_ZONA else None,
                               "c": int(np.searchsorted(cortes, m)) if m and n >= MIN_ZONA and len(cortes) else -1}
        salida[nivel] = gj
        if nivel == "comunas":
            salida["cortes"] = [round(x / 1e6, 1) for x in cortes]
    return salida


def _vecinos(ciudad: str, p: pd.DataFrame) -> dict | None:
    """Los municipios vecinos (data/geo/<ciudad>_vecinos.geojson) donde caen puntos de la ciudad, con cuántos."""
    import shapely
    from shapely.geometry import shape
    if ciudad in zonas.AREAS or not zonas.archivo("vecinos", ciudad).exists():
        return None
    gj = zonas.cargar("vecinos", ciudad)
    i_pt, i_pol = shapely.STRtree([shape(f["geometry"]) for f in gj["features"]]).query(
        shapely.points(p["longitud"], p["latitud"]), predicate="within")
    n = np.bincount(i_pol, minlength=len(gj["features"]))
    gj["features"] = [f | {"properties": {"nombre": f["properties"]["nombre"], "n": int(k)}}
                      for f, k in zip(gj["features"], n) if k >= MIN_VECINO]
    return gj


def generar(ciudad: str, fecha: str) -> dict:
    """Escribe la página, sus puntos y sus zonas; devuelve el resumen para el índice."""
    s = zonas.slug(ciudad)
    p, inm = puntos(ciudad)
    tipos = sorted(p["tipo"].dropna().unique(), key=lambda t: (list(GRUPOS).index(t) if t in GRUPOS else 99, t))
    indice = {t: i for i, t in enumerate(tipos)}
    redondo = lambda x, d=0: None if pd.isna(x) else (round(float(x), d) if d else int(round(float(x))))
    filas = [[round(r.latitud, 5), round(r.longitud, 5), indice[r.tipo], redondo(r.pf / 1e6 if pd.notna(r.pf) else None, 1),
              redondo(r.pm / 1e6 if pd.notna(r.pm) else None, 1), redondo(r.m2), redondo(r.habitaciones),
              _ruta(r.uf, DOMINIOS["fr"]) or None, _ruta(r.um, DOMINIOS["m2"]) or None]
             for r in p.itertuples()]
    lat, lon = p["latitud"], p["longitud"]
    caja = [[round(lat.quantile(.03), 4), round(lon.quantile(.03), 4)], [round(lat.quantile(.97), 4), round(lon.quantile(.97), 4)]]
    z = _zonas(ciudad, inm) if zonas.tiene_zonas(ciudad) else {}
    datos = SALIDA / "datos"
    datos.mkdir(parents=True, exist_ok=True)
    (datos / f"{s}.json").write_text(json.dumps({"tipos": tipos, "grupos": [GRUPOS.get(t, "Otros") for t in tipos],
                                                 "puntos": filas}, separators=(",", ":"), ensure_ascii=False),
                                     encoding="utf-8")
    if (v := _vecinos(ciudad, p)) is not None:
        z["vecinos"] = v
    for nivel in ("comunas", "barrios", "vecinos"):
        if nivel in z:
            (datos / f"{s}_{nivel}.geojson").write_text(json.dumps(z[nivel], separators=(",", ":"), ensure_ascii=False),
                                                       encoding="utf-8")
    dos = int((p["pf"].notna() & p["pm"].notna()).sum())
    area = next((a for a, (n, *_) in zonas.AREAS.items() if n == ciudad), None)
    config = {"slug": s, "ciudad": ciudad, "fecha": fecha, "caja": caja, "division": zonas.division(ciudad),
              "barrios": "barrios" in z, "comunas": "comunas" in z, "cortes": z.get("cortes", []), "rampa": RAMPA,
              "dominios": DOMINIOS, "n": len(filas), "dos": dos, "vecinos": "vecinos" in z,
              "area": {"nombre": area, "slug": zonas.slug(area)} if area else None}
    (SALIDA / f"{s}.html").write_text(PAGINA.replace("__CONFIG__", json.dumps(config, ensure_ascii=False))
                                      .replace("__TITULO__", html.escape(ciudad)), encoding="utf-8")
    print(f"✅ {ciudad}: {len(filas):,} puntos ({dos:,} en los dos portales) → {(SALIDA / f'{s}.html').relative_to(BASE_DIR)}")
    return {"ciudad": ciudad, "slug": s, "n": len(filas), "area": ciudad in zonas.AREAS}


def indice(resumen: list[dict], fecha: str) -> None:
    filas = "".join(
        f'<li><a href="{r["slug"]}.html">{html.escape(r["ciudad"])}</a><span>{r["n"]:,} avisos</span></li>'
        .replace(f'{r["n"]:,}', f'{r["n"]:,}'.replace(",", "."))
        for r in resumen)
    (SALIDA / "index.html").write_text(INDICE.replace("__FILAS__", filas).replace("__FECHA__", fecha), encoding="utf-8")


def ultimo_barrido() -> str:
    from src.consolidar import rutas
    meta = rutas("venta").meta
    return json.loads(meta.read_text())["ultimo_barrido"] if meta.exists() else ""


PAGINA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mapa de mercado · __TITULO__</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.5.3/MarkerCluster.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.5.3/MarkerCluster.Default.min.css">
<style>
  :root { --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --line:#e1e0d9; --plane:#f9f9f7; --surface:#ffffff;
          --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --gray:#898781; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
          --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --plane:#1a1a19; --surface:#0d0d0d; } }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; background: var(--surface); color: var(--ink);
               font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 14px; }
  .barra { position: absolute; z-index: 1000; top: 10px; left: 10px; right: 10px; max-width: 560px;
           background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px;
           box-shadow: 0 2px 10px rgba(0,0,0,.08); }
  h1 { font-size: 16px; margin: 0 0 2px; font-weight: 600; }
  .sub { color: var(--ink-2); font-size: 12.5px; margin: 0 0 8px; }
  .sub a { color: var(--blue); font-weight: 600; text-decoration: none; }
  .filtros { display: flex; flex-wrap: wrap; gap: 6px; }
  .filtros label { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line);
                   border-radius: 99px; padding: 3px 10px; cursor: pointer; font-size: 12.5px; user-select: none; }
  .filtros input { margin: 0; }
  .filtros i { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  #mapa { position: absolute; inset: 0; }
  .leyenda { position: absolute; z-index: 1000; bottom: 18px; left: 10px; background: var(--surface);
             border: 1px solid var(--line); border-radius: 10px; padding: 8px 10px; font-size: 12px; color: var(--ink-2); }
  .leyenda b { color: var(--ink); font-weight: 600; display: block; margin-bottom: 4px; }
  .leyenda span { display: inline-flex; align-items: center; gap: 4px; margin-right: 8px; }
  .leyenda i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .leaflet-popup-content { font-size: 13px; line-height: 1.45; color: #0b0b0b; }
  .leaflet-popup-content a { color: #1c5cab; font-weight: 600; text-decoration: none; }
  .marker-cluster-small, .marker-cluster-medium, .marker-cluster-large { background-color: rgba(42,120,214,.28); }
  .marker-cluster-small div, .marker-cluster-medium div, .marker-cluster-large div {
    background-color: rgba(28,92,171,.88); color: #fff; font-weight: 600; }
  .pie { position: absolute; z-index: 1000; bottom: 18px; right: 10px; font-size: 11px; color: var(--muted);
         background: var(--surface); border-radius: 6px; padding: 2px 6px; }
  @media (max-width: 600px) { .leyenda { font-size: 11px; } h1 { font-size: 15px; } }
</style></head><body>
<div id="mapa"></div>
<div class="barra">
  <h1>Mapa de mercado · __TITULO__</h1>
  <p class="sub" id="sub">Cargando…</p>
  <div class="filtros" id="filtros"></div>
</div>
<div class="leyenda" id="leyenda"></div>
<div class="pie">Avisos públicos de FincaRaíz y Metrocuadrado · no es un avalúo</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.5.3/leaflet.markercluster.min.js"></script>
<script>
const C = __CONFIG__;
const COLOR = {"Vivienda": "#2a78d6", "Comercial": "#eb6834", "Lotes y campo": "#1baf7a", "Otros": "#898781"};
const mapa = L.map("mapa", {preferCanvas: true, zoomControl: false}).fitBounds(C.caja);
L.control.zoom({position: "bottomright"}).addTo(mapa);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  {attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>', maxZoom: 19}).addTo(mapa);
const mill = x => x == null ? "" : (x >= 100 ? Math.round(x).toLocaleString("es-CO") : x.toLocaleString("es-CO", {maximumFractionDigits: 1})) + " M";

// Zonas: comunas o localidades al alejarse, barrios al acercarse
const capas = {};
const estilo = f => ({color: "#ffffff", weight: 1, fillOpacity: f.properties.c >= 0 ? .45 : .15,
                      fillColor: f.properties.c >= 0 ? C.rampa[f.properties.c] : "#c3c2b7"});
const info = f => `<b>${f.properties.nombre}</b>${f.properties.zona ? " · " + f.properties.zona : ""}<br>` +
  (f.properties.m ? `${f.properties.n.toLocaleString("es-CO")} viviendas · mediana <b>${f.properties.m.toLocaleString("es-CO")} M/m²</b>`
                  : `pocas viviendas para dar un precio`);
function zonas(nivel) {
  return fetch(`datos/${C.slug}_${nivel}.geojson`).then(r => r.json()).then(gj => {
    capas[nivel] = L.geoJSON(gj, {style: estilo, onEachFeature: (f, l) => l.bindTooltip(info(f), {sticky: true})});
  });
}
function alternar() {
  const z = mapa.getZoom(), cerca = C.barrios && z >= 14;
  for (const [nivel, capa] of Object.entries(capas)) {
    const ver = nivel === "barrios" ? cerca : !cerca;
    if (ver && !mapa.hasLayer(capa)) capa.addTo(mapa).bringToBack();
    if (!ver && mapa.hasLayer(capa)) mapa.removeLayer(capa);
  }
}
Promise.all([C.comunas ? zonas("comunas") : null, C.barrios ? zonas("barrios") : null]).then(() => { alternar(); mapa.on("zoomend", alternar); });

// Municipios vecinos donde caen avisos que dicen ser de la ciudad (Yumbo, Jamundí…): en gris, siempre visibles
const enArea = C.area ? `<br>Véalos con sus zonas en el mapa de ${C.area.nombre}` : "";
if (C.vecinos) fetch(`datos/${C.slug}_vecinos.geojson`).then(r => r.json()).then(gj =>
  L.geoJSON(gj, {style: {color: "#52514e", weight: 1.5, dashArray: "5 4", fillColor: "#898781", fillOpacity: .2},
    onEachFeature: (f, l) => l.bindTooltip(`<b>${f.properties.nombre}</b> · otro municipio<br>` +
      `${f.properties.n.toLocaleString("es-CO")} avisos anunciados en ${C.ciudad} caen aquí${enArea}`, {sticky: true})
  }).addTo(mapa).bringToBack());

// Leyenda del color de las zonas
const lims = C.cortes;
document.getElementById("leyenda").innerHTML = lims.length ?
  `<b>Precio mediano por m² de la vivienda</b>` + C.rampa.map((c, i) =>
    `<span><i style="background:${c}"></i>${i === 0 ? "< " + lims[0] : i === 4 ? "> " + lims[3] : lims[i - 1] + "–" + lims[i]} M</span>`).join("") +
  `<br><span>Al acercarse: barrios. Toque una zona o un punto para ver el detalle.</span>` : "";

// Puntos: cargados al abrir, agrupados hasta acercarse
const grupo = L.markerClusterGroup({chunkedLoading: true, maxClusterRadius: 45, disableClusteringAtZoom: 17,
                                    showCoverageOnHover: false});
mapa.addLayer(grupo);
let marcas = [], datos;
function popup(p) {
  const tipo = datos.tipos[p[2]], [pf, pm, m2, hab, uf, um] = p.slice(3);
  const precio = pf ?? pm, pm2 = precio && m2 ? ` · ${(precio / m2).toLocaleString("es-CO", {maximumFractionDigits: 1})} M/m²` : "";
  let h = `<b>${tipo}</b>${m2 ? " · " + m2 + " m²" : ""}${hab ? " · " + hab + " hab" : ""}<br><b>${mill(precio)}</b>${pm2}<br>`;
  if (uf) h += `FincaRaíz ${pf && pm && pf !== pm ? mill(pf) + " " : ""}<a href="${C.dominios.fr}${uf}" target="_blank" rel="noopener">ver ↗</a><br>`;
  if (um) h += `Metrocuadrado ${pf && pm && pf !== pm ? mill(pm) + " " : ""}<a href="${C.dominios.m2}${um}" target="_blank" rel="noopener">ver ↗</a>`;
  return h;
}
function pintar() {
  const activos = new Set([...document.querySelectorAll("#filtros input:checked")].map(i => i.value));
  grupo.clearLayers();
  const visibles = marcas.filter(m => activos.has(m.grupo));
  grupo.addLayers(visibles);
  document.getElementById("sub").innerHTML =
    `${visibles.length.toLocaleString("es-CO")} de ${marcas.length.toLocaleString("es-CO")} avisos activos · ` +
    `${C.dos.toLocaleString("es-CO")} en los dos portales · ${C.fecha}` +
    (C.area ? ` · <a href="${C.area.slug}.html">${C.area.nombre} ↗</a>` : "");
}
fetch(`datos/${C.slug}.json`).then(r => r.json()).then(d => {
  datos = d;
  marcas = d.puntos.map(p => {
    const g = d.grupos[p[2]], ambos = p[7] && p[8];
    const m = L.circleMarker([p[0], p[1]], {radius: 5, weight: ambos ? 2 : 1, color: ambos ? "#0b0b0b" : "#ffffff",
                                            fillColor: COLOR[g] || COLOR.Otros, fillOpacity: .9});
    m.grupo = g;
    m.on("click", () => m.bindPopup(popup(p)).openPopup());
    return m;
  });
  const cuenta = {};
  marcas.forEach(m => cuenta[m.grupo] = (cuenta[m.grupo] || 0) + 1);
  document.getElementById("filtros").innerHTML = Object.keys(COLOR).filter(g => cuenta[g]).map(g =>
    `<label><input type="checkbox" value="${g}" checked><i style="background:${COLOR[g]}"></i>${g} · ${cuenta[g].toLocaleString("es-CO")}</label>`).join("") +
    `<label><i style="background:#fff;border:2px solid #0b0b0b"></i>borde oscuro: en los dos portales</label>`;
  document.querySelectorAll("#filtros input").forEach(i => i.addEventListener("change", pintar));
  pintar();
});
</script></body></html>"""

INDICE = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mapas de mercado</title>
<style>
  :root { --ink:#0b0b0b; --ink-2:#52514e; --line:#e1e0d9; --surface:#ffffff; --blue:#1c5cab; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
          --ink:#f0efec; --ink-2:#c3c2b7; --line:#2c2c2a; --surface:#0d0d0d; --blue:#86b6ef; } }
  body { margin: 0; background: var(--surface); color: var(--ink); font-family: Inter, system-ui, sans-serif; }
  main { max-width: 640px; margin: 0 auto; padding: 28px 16px; }
  h1 { font-size: 22px; margin: 0 0 4px; } p { color: var(--ink-2); margin: 0 0 18px; }
  ul { list-style: none; padding: 0; margin: 0; }
  li { display: flex; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 12px 0; }
  a { color: var(--blue); font-weight: 600; text-decoration: none; } span { color: var(--ink-2); }
</style></head><body><main>
<h1>Mapas de mercado</h1>
<p>Avisos activos de vivienda, comercio y lotes en FincaRaíz y Metrocuadrado · __FECHA__</p>
<ul>__FILAS__</ul>
</main></body></html>"""


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Mapa interactivo por ciudad (E6)")
    a.add_argument("--ciudad")
    a.add_argument("--todas", action="store_true", help="las ciudades y áreas de src/zonas.py, y el índice")
    args = a.parse_args()
    fecha = ultimo_barrido()
    lista = list(zonas.CIUDADES) if args.todas else [args.ciudad or "Cali"]
    resumen = [generar(c, fecha) for c in lista]
    if args.todas:
        indice(resumen, fecha)
