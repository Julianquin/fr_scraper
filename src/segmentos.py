"""
Árbol de segmentos para barrer FincaRaíz completo (bloques E1 y E3). SOLO LECTURA: pide la
página 1 de cada segmento para leer su total; no barre nada. Sirve igual para venta y arriendo
(la raíz de la URL: /venta/… o /arriendo/…).

El buscador solo deja recorrer 10.000 resultados por consulta, así que el país se parte en
segmentos públicos (URLs normales del sitio) que quepan, con margen, en ese tope:

    Departamento (33, lista fija)                                    /<raíz>/<depto>
      └─ si no cabe → Tipo de inmueble (15, lista fija del portal)   /<raíz>/<tipo>/<depto>
           └─ si no cabe → Ciudad (descubierta en los avisos)        /<raíz>/<tipo>/<ciudad>/<depto>
                └─ si no cabe → Zona (descubierta en los avisos)     /<raíz>/<tipo>/<zona>/<ciudad>

Al partir un segmento, compara la suma de sus hijos con su total: la diferencia ("hueco") son
avisos que ningún hijo cubre (sin tipo, ciudad o zona reconocible). El resultado dice cuántas
páginas cuesta un barrido completo y qué cobertura nacional lograría.

Las ciudades y las zonas se descubren mirando una muestra de avisos, y una muestra puede no ver
una zona o un municipio con pocos avisos etiquetados: del 5 al 11-nov la zona Chapinero de Bogotá
(1.765 apartamentos) quedó fuera del barrido, ~800 apartamentos se dieron por retirados y volvieron
el 12-nov. Dos defensas: ZONAS_CONOCIDAS (las zonas de Bogotá y Medellín se barren siempre) y
`recordar` (cualquier segmento del plan de ayer que la muestra de hoy no vio se añade igual, salvo
que ya lo cubra otro segmento de hoy).

Uso:
    python -m src.segmentos [--raiz arriendo]   # arma el árbol → data/analisis/e1/segmentos_arbol[_arriendo].csv
"""
from __future__ import annotations
import argparse
import math
import time
import unicodedata

import pandas as pd
import requests

from src.config import BASE_DIR
from src.fincaraiz_http import POR_PAGINA, ULTIMA_PAGINA, pedir

LIMITE = 9_500                   # margen bajo el tope de 10.000 para que el crecimiento no lo rompa
# Zonas de las ciudades que se parten por zona (slug corto del portal → nombre), vistas el 13-nov-2026: se barren
# siempre, aunque la muestra de la noche no las vea (las que no tengan avisos de ese tipo se saltan solas)
ZONAS_CONOCIDAS = {
    "bogota": {"zona-centro": "Zona centro", "zona-chapinero": "Zona chapinero", "zona-noroccidental": "Zona Noroccidente",
               "zona-nororiental": "Zona nororiental", "zona-norte": "Zona norte", "zona-occidental": "Zona occidental",
               "zona-sur": "Zona sur"},
    "medellin": {"centro": "Centro", "noroccidente": "Noroccidente", "nororiente": "Nororiente",
                 "occidente": "Occidente", "suroccidente": "Suroccidente", "suroriente": "Suroriente"},
}
OUT = BASE_DIR / "data" / "analisis" / "e1"
RAICES = ("venta", "arriendo")

DEPARTAMENTOS = [
    "amazonas", "antioquia", "arauca", "atlantico", "bogota-dc", "bolivar", "boyaca", "caldas",
    "caqueta", "casanare", "cauca", "cesar", "choco", "cordoba", "cundinamarca", "guainia",
    "guaviare", "huila", "la-guajira", "magdalena", "meta", "narino", "norte-de-santander",
    "putumayo", "quindio", "risaralda", "san-andres", "santander", "sucre", "tolima",
    "valle-del-cauca", "vaupes", "vichada",
]
# Lista oficial del portal (Filter:propertyType), en plural como va en la URL
TIPOS = ["casas", "apartamentos", "apartaestudios", "cabanas", "casas-campestres", "casas-lotes",
         "fincas", "habitaciones", "lotes", "bodegas", "consultorios", "locales", "oficinas",
         "parqueaderos", "edificios"]
# Para descubrir ciudades/zonas se muestrean páginas repartidas en varios órdenes
ORDENES_MUESTRA = [3, 1, 0, 4, 5]
PAGINAS_MUESTRA = [1, 60, 150, 250, 350, ULTIMA_PAGINA]


def slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return "-".join(t.replace(",", " ").replace(".", " ").split())


class Arbol:
    def __init__(self, pausa: float, raiz: str = "venta"):
        self.s = requests.Session()
        self.pausa = pausa
        self.raiz = raiz
        self.filas: list[dict] = []
        self.huecos: list[dict] = []
        self.descartados: list[str] = []
        self._totales: dict[str, int] = {}
        self._estados: dict[str, list[str]] = {}     # departamento de cada aviso de la página 1

    def total(self, ruta: str) -> int:
        if ruta not in self._totales:
            try:
                data, info, _ = pedir(self.s, ruta, 1)
                self._totales[ruta] = int(info.get("total") or 0)
                self._estados[ruta] = [((a.get("locations") or {}).get("state") or [{}])[0].get("name", "")
                                       for a in data]
            except Exception:  # noqa: BLE001  ruta inexistente → 0
                self._totales[ruta] = 0
            time.sleep(self.pausa)
        return self._totales[ruta]

    def es_del_departamento(self, ruta: str, d: str) -> bool:
        """¿La mayoría de los avisos de `ruta` son del departamento `d`? Algunos avisos vienen mal
        etiquetados (p. ej. Fusagasugá dentro de Bogotá D.C.) y la URL de la ciudad devuelve igual
        sus avisos; esos ya los cubre el segmento de su departamento real."""
        clave = d.replace("-", "")
        estados = [slug(e).replace("-", "") for e in self._estados.get(ruta, []) if e]
        return bool(estados) and sum(clave in e or e in clave for e in estados) >= len(estados) / 2

    def muestrear(self, ruta: str, nivel: str) -> dict[str, str]:
        """Nombres de `nivel` ('city' | 'zone') vistos en páginas repartidas de `ruta` → {nombre: slug_portal}."""
        vistos: dict[str, str] = {}
        ultima = min(ULTIMA_PAGINA, math.ceil(self.total(ruta) / POR_PAGINA))
        for orden in ORDENES_MUESTRA:
            for n in sorted({min(p, ultima) for p in PAGINAS_MUESTRA}):
                try:
                    data, _, _ = pedir(self.s, ruta, n, orden=orden)
                except Exception:  # noqa: BLE001
                    continue
                for a in data:
                    for v in (a.get("locations") or {}).get(nivel) or []:
                        vistos.setdefault(v.get("name", ""), (v.get("slug") or [""])[0])
                time.sleep(self.pausa)
        return vistos

    def hoja(self, ruta: str, nivel: str, padre: str | None, total: int, accion: str = "barrer") -> None:
        self.filas.append({"ruta": ruta, "nivel": nivel, "padre": padre, "total": total,
                           "paginas": min(ULTIMA_PAGINA, math.ceil(total / POR_PAGINA)), "accion": accion})

    def partido(self, ruta: str, nivel: str, padre: str | None, total: int, hijos_suma: int) -> None:
        self.filas.append({"ruta": ruta, "nivel": nivel, "padre": padre, "total": total,
                           "paginas": 0, "accion": "partido"})
        self.huecos.append({"segmento": ruta, "total": total, "hijos": hijos_suma,
                            "hueco": total - hijos_suma})

    # ── Niveles ──────────────────────────────────────────────────────────────
    def departamento(self, d: str) -> int:
        ruta = f"{self.raiz}/{d}"
        t = self.total(ruta)
        if t == 0:
            return 0
        if t <= LIMITE:
            self.hoja(ruta, "departamento", None, t)
            return t
        suma = sum(self.tipo(tp, d, ruta) for tp in TIPOS)
        self.partido(ruta, "departamento", None, t, suma)
        return min(t, suma)

    def tipo(self, tp: str, d: str, padre: str) -> int:
        ruta = f"{self.raiz}/{tp}/{d}"
        t = self.total(ruta)
        if t == 0:
            return 0
        if t <= LIMITE:
            self.hoja(ruta, "tipo", padre, t)
            return t
        ciudades = self.muestrear(ruta, "city")
        suma = 0
        for nombre in sorted(ciudades):
            suma += self.ciudad(tp, slug(nombre), d, ruta)
        self.partido(ruta, "tipo", padre, t, suma)
        return min(t, suma)

    def ciudad(self, tp: str, c: str, d: str, padre: str) -> int:
        ruta = f"{self.raiz}/{tp}/{c}/{d}"
        t = self.total(ruta)
        if t == 0:
            return 0
        if not self.es_del_departamento(ruta, d):
            self.descartados.append(ruta)
            return 0
        if t <= LIMITE:
            self.hoja(ruta, "ciudad", padre, t)
            return t
        # slug del portal: 'zone-colombia-bogota-zona-norte' → 'zona-norte'
        vistas = {(sp.split(f"-{c}-", 1)[-1] if f"-{c}-" in sp else slug(n)): n
                  for n, sp in self.muestrear(ruta, "zone").items()}
        conocidas = ZONAS_CONOCIDAS.get(c, {})
        if nuevas := set(vistas) - set(conocidas):
            print(f"ℹ️  {ruta}: zonas que no están en ZONAS_CONOCIDAS: {sorted(nuevas)} (añadirlas)")
        suma = 0
        for z in sorted(set(vistas) | set(conocidas)):
            rz = f"{self.raiz}/{tp}/{z}/{c}"
            tz = self.total(rz)
            if tz == 0:
                continue
            # Una zona que no cabe se barre en los dos órdenes de precio (hasta 2 × 10.000)
            self.hoja(rz, "zona", ruta, tz, "barrer" if tz <= LIMITE else "barrer_dos_ordenes")
            suma += tz
        self.partido(ruta, "ciudad", padre, t, suma)
        return min(t, suma)


def recordar(arbol: "Arbol", previo: pd.DataFrame) -> list[str]:
    """Añade al árbol de hoy los segmentos hoja del plan anterior (`previo`, su plan.csv) que la muestra de hoy no
    vio: los que no están en el árbol de hoy (ni como hoja ni partidos) y cuyo padre no es hoy una hoja (si lo es,
    ya los cubre). Solo si todavía tienen avisos. Devuelve las rutas añadidas."""
    hoy = {f["ruta"] for f in arbol.filas}
    hojas_hoy = {f["ruta"] for f in arbol.filas if f["accion"] != "partido"}
    añadidas = []
    for _, h in previo.iterrows():
        if h["ruta"] in hoy or (isinstance(h.get("padre"), str) and h["padre"] in hojas_hoy):
            continue
        t = arbol.total(h["ruta"])
        if t:
            arbol.hoja(h["ruta"], h["nivel"], h.get("padre") if isinstance(h.get("padre"), str) else None, t,
                       "barrer" if t <= LIMITE else "barrer_dos_ordenes")
            añadidas.append(h["ruta"])
    return añadidas


def construir(pausa: float, raiz: str = "venta", previo: pd.DataFrame | None = None) -> pd.DataFrame:
    """El árbol de hoy; con `previo` (el plan de ayer), más los segmentos de ayer que la muestra de hoy no vio."""
    arbol = Arbol(pausa, raiz)
    t0 = time.time()
    nacional = arbol.total(raiz)
    cubiertos = sum(arbol.departamento(d) for d in DEPARTAMENTOS)
    if previo is not None and len(previo):
        if añadidas := recordar(arbol, previo):
            print(f"⚠️  La muestra de hoy no vio {len(añadidas)} segmentos del plan anterior; se barren igual: {añadidas}")
    df = pd.DataFrame(arbol.filas)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / ("segmentos_arbol.csv" if raiz == "venta" else f"segmentos_arbol_{raiz}.csv"), index=False)

    hojas = df[df["accion"] != "partido"]
    solape = hojas["total"].sum() - cubiertos
    print(f"Árbol armado en {(time.time() - t0) / 60:.1f} min · {len(arbol._totales)} consultas")
    print(f"Segmentos a barrer: {len(hojas)} · páginas por barrido: {hojas['paginas'].sum():,} "
          f"(~{hojas['paginas'].sum() * 1.4 / 3600:.1f} h en secuencia; ~{hojas['paginas'].sum() * 1.4 / 3600 / 4:.1f} h con 4 jobs)")
    print(f"Cobertura (conservadora): {cubiertos:,} de {nacional:,} avisos nacionales ({cubiertos / nacional:.1%}) · "
          f"solapamiento entre segmentos: {solape:,} avisos que se recorrerían dos veces")
    if arbol.descartados:
        print(f"Ciudades descartadas por ser de otro departamento ({len(arbol.descartados)}):", arbol.descartados)
    print("Segmentos por nivel:", hojas["nivel"].value_counts().to_dict())
    if (hojas["accion"] == "barrer_dos_ordenes").any():
        print("⚠️ Zonas que no caben (se barren en dos órdenes):",
              hojas.loc[hojas["accion"] == "barrer_dos_ordenes", "ruta"].tolist())
    print("\nHuecos al partir (avisos que ningún hijo cubre):")
    for h in arbol.huecos:
        tipo = "hueco" if h["hueco"] >= 0 else "solape"
        print(f"   {h['segmento']:38} total {h['total']:>6} · hijos {h['hijos']:>6} · {tipo} {abs(h['hueco']):>5} "
              f"({abs(h['hueco']) / h['total']:.1%})")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Arma el árbol de segmentos de FincaRaíz (solo lectura)")
    p.add_argument("--pausa", type=float, default=0.6)
    p.add_argument("--raiz", choices=RAICES, default="venta")
    a = p.parse_args()
    construir(a.pausa, a.raiz)
