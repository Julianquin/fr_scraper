"""
Barrido diario de Metrocuadrado por HTTP (bloque E8): vivienda en venta de todo el país. Corre DESPUÉS
de los de FincaRaíz y no toca nada de ellos: escribe solo su propio crudo y su propio estado.
SOLO páginas públicas de listado, con pausas de cortesía; nada de APIs internas. El robots.txt permite
/venta/ y prohíbe /detail/: no se visita el detalle.

Lo que hay (exploración del 27-oct-2026):
  - Los avisos vienen en el HTML (Next.js, bloques `self.__next_f.push`) dentro de `initialResults`:
    totalHits (total del filtro) y los primeros ~50 resultados (+ algunos destacados).
  - La paginación (?from=50…) NO la resuelve el servidor: cada URL entrega solo su primera página.
  - Filtros públicos en la RUTA (los de ?query no los aplica el servidor), combinables y en cualquier orden:
    /{tipo}/venta/{ciudad | dep-departamento}/{barrio}/{n}-habitaciones/{n}-banos/estrato-{e}/{n}-millones/
  - `{n}-millones` filtra «hasta n millones» Y ORDENA POR PRECIO DE MAYOR A MENOR: cada página trae los 50
    más caros bajo el techo. Bajando el techo se recorre todo (la caminata).
  - `dep-…` busca el departamento por coincidencia de texto: "cauca" incluye Valle del Cauca, "santander"
    incluye Norte de Santander y "cundinamarca" incluye Bogotá. Con DEPARTAMENTOS (sin los repetidos) la
    suma da exacto el total nacional (92.199 apartamentos el 27-oct).
  - Si un filtro deja 0 resultados, el servidor lo relaja y muestra un total grande SIN avisos: es un
    segmento vacío, no uno incompleto.
  - No hay fecha de publicación: consolidar cuenta como alta lo que aparece desde el barrido anterior.
  - Trae el teléfono y el WhatsApp del anunciante: NO se guardan (`aplanar` no los toma).

Cómo se recorre: el país se parte en unidades tipo × departamento (los departamentos grandes, además, por
habitaciones). Cada unidad se camina por precio: el techo baja al precio más bajo de cada página. Cuando
más de 50 avisos caen en el mismo escalón de un millón (p. ej. 250 millones justos), ese escalón se parte
por habitaciones, baños y estrato (EMPATES); lo que no se resuelve ni así queda en `sin_resolver`.
Cali, 27-oct: 99,2% de cobertura en 496 páginas (~3,5 s por página con la pausa).

Tres pasos, como el barrido de FincaRaíz (en el CI, .github/workflows/metrocuadrado.yml):
    python -m src.metrocuadrado_http plan --grupos 4        # totales → unidades repartidas en grupos
    python -m src.metrocuadrado_http barrer --grupo 0       # camina las unidades de un grupo
    python -m src.metrocuadrado_http juntar [--subir]       # une, deduplica, mide cobertura, sube a R2

Archivos de trabajo en data/http_m2_venta/ y crudo en data/raw_http_m2_venta/<fecha>/m2_venta.parquet
(en R2: raw/m2_venta_http/fecha=<fecha>/scrape.parquet y reporte.json). El estado (altas, bajas,
cambios de precio) lo lleva src/consolidar.py con --raiz m2_venta.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import time

import pandas as pd
import requests

from src.fincaraiz_http import NUMERICOS, UA, guardar_parquet
from src.scraper_http import carpetas, hoy

RAIZ = "m2_venta"
BASE = "https://www.metrocuadrado.com"
POR_PAGINA = 50
TIPOS = {"apartamentos": "Apartamento", "casas": "Casa"}     # vivienda (lo que usan la alerta y el modelo)
DEPARTAMENTOS = ["amazonas", "antioquia", "arauca", "atlantico", "bolivar", "boyaca", "caldas", "caqueta",
                 "casanare", "cauca", "cesar", "choco", "cordoba", "cundinamarca", "guainia", "guaviare", "huila",
                 "la-guajira", "magdalena", "meta", "narino", "putumayo", "quindio", "risaralda", "san-andres",
                 "santander", "sucre", "tolima", "vaupes", "vichada"]
GRANDE = 3_000            # una unidad con más avisos se parte por habitaciones (para repartir mejor los grupos)
HABITACIONES = [1, 2, 3, 4, 5]                               # «5-habitaciones» incluye 5 o más
TECHO = 100_000                                              # millones: el primer techo, sobre cualquier aviso
EMPATES = (("{}-habitaciones", HABITACIONES), ("{}-banos", [1, 2, 3, 4, 5]), ("estrato-{}", [1, 2, 3, 4, 5, 6]))
_BLOQUE = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', re.S)


def ruta(tipo: str, zona: str = "", *filtros: str) -> str:
    """URL pública de listado: ruta("apartamentos", "dep-antioquia", "3-habitaciones", "250-millones")."""
    return "/" + "/".join(p for p in [tipo, "venta", zona, *filtros] if p) + "/"


def _objeto(payload: str, clave: str) -> dict | None:
    """El objeto JSON que sigue a `"clave":` dentro del payload."""
    k = payload.find(f'"{clave}"')
    if k < 0:
        return None
    return json.JSONDecoder().raw_decode(payload, payload.find("{", k))[0]   # respeta llaves dentro de textos


def pedir(sesion: requests.Session, ruta_url: str, reintentos: int = 3) -> tuple[int, list[dict]]:
    """(total del filtro, avisos de la primera página) de una URL pública de listado."""
    for intento in range(1, reintentos + 1):
        try:
            r = sesion.get(BASE + ruta_url, headers=UA, timeout=60)
            r.raise_for_status()
            payload = "".join(json.loads('"' + p + '"') for p in _BLOQUE.findall(r.text))
            d = _objeto(payload, "initialResults")
            if d is None:
                raise ValueError(f"Sin resultados en {ruta_url}")
            return int(d.get("totalHits") or 0), d.get("results") or []
        except (requests.RequestException, ValueError):
            if intento == reintentos:
                raise
            time.sleep(5 * intento)
    raise RuntimeError("inalcanzable")


def _caracteristicas(a: dict) -> dict[str, str]:
    out = {}
    for f in a.get("featured") or []:
        k, _, v = f.strip().partition(":")
        out[k.strip()] = v.strip()
    return out


def aplanar(a: dict) -> dict:
    """Un aviso de Metrocuadrado → el esquema del barrido de FincaRaíz (+ `portal`, `tipo_anunciante`,
    `antiguedad_texto`). Sin teléfonos ni WhatsApp."""
    c = _caracteristicas(a)
    datos = a.get("data") or {}
    loc = a.get("localizacion") or {}
    etiquetas = [e for e in (a.get("mestadoinmueble"), "Proyecto" if a.get("mnombreproyecto") or a.get("badge") else None,
                             a.get("tipovivienda")) if e]
    fila = {
        "id_inmueble": str(a.get("midinmueble")),
        "url": BASE + (a.get("link") or ""),
        "tipo": (a.get("mtipoinmueble") or {}).get("nombre"),
        "precio": a.get("mvalorventa"),
        "administracion": datos.get("mvaloradministracion"),
        "fecha_publicacion": None,
        "fecha_actualizacion": None,
        "habitaciones": a.get("mnrocuartos"),
        "banos": a.get("mnrobanos"),
        "parqueaderos": a.get("mnrogarajes"),
        "m2": a.get("marea"),
        "m2_construido": a.get("mareac"),
        "m2_terreno": c.get("areaLote"),
        "estrato": a.get("estrato"),
        "piso": c.get("nroPiso"),
        "antiguedad": None,
        "estado": a.get("mestadoinmueble"),
        "latitud": loc.get("lat"),
        "longitud": loc.get("lon"),
        "departamento": None,
        "ciudad": (a.get("mciudad") or {}).get("nombre"),
        "localidad": None,
        "comuna": None,
        "zona": (a.get("mzona") or {}).get("nombre"),
        "barrio": (a.get("mnombrecomunbarrio") or a.get("mbarrio") or "").strip() or None,
        "direccion": None,
        "anunciante": datos.get("mnombrevisitor"),
        "anunciante_id": a.get("midempresa"),
        "codigo_anunciante": None,
        "destacado": 1 if a.get("badge") else 0,
        "etiquetas": json.dumps(etiquetas, ensure_ascii=False),
        "amenidades": "|".join(k for k, v in c.items() if v == "S") or None,
        "n_imagenes": len(a.get("mgaleriainmueble") or []),
        "descripcion": a.get("comment"),
        "portal": "metrocuadrado",
        "barrio_catalogo": (a.get("mbarrio") or "").strip() or None,
        "tipo_anunciante": a.get("OwnerType"),
        "antiguedad_texto": c.get("tiempoConstruido"),
    }
    for k in NUMERICOS:
        try:
            fila[k] = float(fila[k]) if fila[k] not in (None, "", "NA") else None
        except (TypeError, ValueError):
            fila[k] = None
    if not (fila["latitud"] and fila["longitud"]):            # ~7% viene con 0,0: sin ubicación
        fila["latitud"] = fila["longitud"] = None
    return fila


class Barrido:
    """Camina por precio las unidades de un grupo (ver el docstring del módulo). Guarda avances cada
    GUARDAR_CADA páginas y, si se relanza el mismo día, retoma donde iba (sirve en local; en el CI cada
    job arranca limpio)."""

    GUARDAR_CADA = 50

    def __init__(self, nombre: str, pausa: float = 1.0):
        self.pausa = pausa
        self.s = requests.Session()
        self.avisos: dict[str, dict] = {}                      # id → fila ya aplanada
        self.paginas: list[dict] = []
        self.sin_resolver: list[str] = []
        self.hechos: dict[str, tuple[int, float | None, list[str]]] = {}
        self.ids: set[str] = set()                             # lo de la unidad en curso
        self.n_paginas = self.errores = 0
        self.avance = carpetas(RAIZ)[0] / f"{nombre}_{hoy()}.avance.json"
        if self.avance.exists():                                         # retomar
            av = json.loads(self.avance.read_text())
            self.avisos = av["avisos"]                                    # sin_resolver se rehace al recorrer
            self.paginas = [g for g in av["paginas"] if g["total"] is not None]   # las fallidas se reintentan
            self.hechos = {g["ruta"]: (g["total"], g["piso"], g["ids"]) for g in self.paginas}
            print(f"↩️  Retomando: {len(self.hechos)} páginas y {len(self.avisos):,} avisos ya leídos", flush=True)
        self.t0 = time.time()

    def guardar_avance(self) -> None:
        self.avance.parent.mkdir(parents=True, exist_ok=True)
        self.avance.write_text(json.dumps({"avisos": self.avisos, "paginas": self.paginas}, ensure_ascii=False))

    def leer(self, r: str) -> tuple[int | None, float | None]:
        """(total del filtro, precio más bajo de la página en millones; None si vino vacía).
        Total None: la página falló. Una página sin avisos vale 0 aunque declare otro total (el servidor
        relajó el filtro)."""
        if r not in self.hechos:
            try:
                total, res = pedir(self.s, r)
            except Exception as e:  # noqa: BLE001  una página caída no frena el barrido
                self.paginas.append({"ruta": r, "total": None, "piso": None, "ids": [], "error": str(e)[:120]})
                self.n_paginas += 1
                self.errores += 1
                return None, None
            finally:
                time.sleep(self.pausa)
            ids = []
            for a in res:
                ids.append(str(a.get("midinmueble")))
                self.avisos.setdefault(ids[-1], aplanar(a))
            precios = [float(a["mvalorventa"]) / 1e6 for a in res if float(a.get("mvalorventa") or 0) > 0]
            pagina = {"ruta": r, "total": total if res else 0, "piso": min(precios) if precios else None, "ids": ids}
            self.paginas.append(pagina)
            self.hechos[r] = (pagina["total"], pagina["piso"], ids)
            if len(self.paginas) % self.GUARDAR_CADA == 0:
                self.guardar_avance()
                print(f"   {len(self.paginas):,} páginas · {len(self.avisos):,} avisos · "
                      f"{(time.time() - self.t0) / 60:.0f} min", flush=True)
        total, piso, ids = self.hechos[r]
        self.ids.update(ids)
        self.n_paginas += 1
        return total, piso

    def caminar(self, tipo: str, zona: str, filtros: tuple[str, ...], techo: int, hasta: int = 0) -> None:
        """Todos los avisos de la unidad con precio en (hasta, techo] millones."""
        while techo > hasta:
            r = ruta(tipo, zona, *filtros, f"{techo}-millones")
            total, piso = self.leer(r)
            if total is None:                          # falló: lo que queda bajo este techo no se leyó
                self.sin_resolver.append(r)
                return
            if total <= POR_PAGINA or piso is None or piso <= hasta:
                return
            siguiente = math.ceil(piso)
            if siguiente >= techo:                     # más de 50 en (techo-1, techo]: partir el escalón
                self.partir(tipo, zona, filtros, techo)
                siguiente = techo - 1
            techo = siguiente

    def partir(self, tipo: str, zona: str, filtros: tuple[str, ...], techo: int) -> None:
        """Parte un escalón de empate por el primer filtro de EMPATES que la unidad aún no use."""
        libres = [(p, v) for p, v in EMPATES if not any(re.fullmatch(p.replace("{}", r"\d+"), f) for f in filtros)]
        if not libres:
            self.sin_resolver.append(ruta(tipo, zona, *filtros, f"{techo}-millones"))
            return
        patron, valores = libres[0]
        for v in valores:
            self.caminar(tipo, zona, (*filtros, patron.format(v)), techo, techo - 1)

    def unidad(self, tipo: str, zona: str, filtros: tuple[str, ...] = ()) -> dict:
        """Recorre una unidad entera: su primera página (trae también los avisos sin precio) y la caminata."""
        self.ids, self.n_paginas, self.errores, antes = set(), 0, 0, len(self.sin_resolver)
        r = ruta(tipo, zona, *filtros)
        total, _ = self.leer(r)
        if total is None or total > POR_PAGINA:
            self.caminar(tipo, zona, filtros, TECHO)
        return {"unidad": r, "declarado": total or 0, "unicos": len(self.ids), "paginas": self.n_paginas,
                "errores": self.errores, "sin_resolver": len(self.sin_resolver) - antes}


# ── 1. Plan ──────────────────────────────────────────────────────────────────
def plan(grupos: int, pausa: float = 1.0) -> pd.DataFrame:
    """Totales por tipo y departamento (y por habitaciones en los grandes) → unidades repartidas en
    `grupos` de carga parecida. La carga es el total declarado: las páginas le son proporcionales."""
    s = requests.Session()

    def total(r: str) -> int:
        n, res = pedir(s, r)
        time.sleep(pausa)
        return n if res else 0                     # sin avisos: el servidor relajó el filtro

    nacional, filas = {}, []
    for tipo in TIPOS:
        nacional[tipo] = total(ruta(tipo))
        for d in DEPARTAMENTOS:
            zona = f"dep-{d}"
            n = total(ruta(tipo, zona))
            if n > GRANDE:
                filas += [{"tipo": tipo, "zona": zona, "filtro": f"{h}-habitaciones",
                           "declarado": total(ruta(tipo, zona, f"{h}-habitaciones"))} for h in HABITACIONES]
            elif n:
                filas.append({"tipo": tipo, "zona": zona, "filtro": "", "declarado": n})
        suma = sum(f["declarado"] for f in filas if f["tipo"] == tipo)
        print(f"{tipo}: {nacional[tipo]:,} en el país · {suma:,} en las unidades", flush=True)
    unidades = pd.DataFrame(filas)
    unidades = unidades[unidades["declarado"] > 0].sort_values("declarado", ascending=False).reset_index(drop=True)
    carga, asignado = [0] * grupos, []
    for n in unidades["declarado"]:                    # al grupo con menos carga (reparto voraz)
        g = carga.index(min(carga))
        asignado.append(g)
        carga[g] += n
    unidades["grupo"] = asignado
    trabajo = carpetas(RAIZ)[0]
    trabajo.mkdir(parents=True, exist_ok=True)
    unidades.to_csv(trabajo / "plan.csv", index=False)
    (trabajo / "nacional.json").write_text(json.dumps(
        {"total_nacional": sum(nacional.values()), "por_tipo": nacional, "fecha": hoy()}))
    print(f"Plan: {len(unidades)} unidades en {grupos} grupos · avisos por grupo: {carga}")
    return unidades


# ── 2. Barrer ────────────────────────────────────────────────────────────────
def barrer(grupo: int, pausa: float = 1.0, solo: int | None = None) -> pd.DataFrame:
    trabajo, _ = carpetas(RAIZ)
    unidades = pd.read_csv(trabajo / "plan.csv", keep_default_na=False)
    unidades = unidades[unidades["grupo"] == grupo]
    if solo:
        unidades = unidades.tail(solo)                 # para pruebas: las unidades más chicas
    b = Barrido(f"grupo{grupo}", pausa)
    informe = []
    for i, u in enumerate(unidades.itertuples(), 1):
        inf = b.unidad(u.tipo, u.zona, (u.filtro,) if u.filtro else ())
        informe.append(inf)
        print(f"[{i}/{len(unidades)}] {inf['unidad']:50} {inf['unicos']:>6}/{inf['declarado']:<6} "
              f"({inf['unicos'] / max(inf['declarado'], 1):.0%}) · {inf['paginas']} págs · {inf['errores']} errores · "
              f"{inf['sin_resolver']} sin resolver · {(time.time() - b.t0) / 60:.0f} min", flush=True)
    b.guardar_avance()
    df = pd.DataFrame(list(b.avisos.values()))
    guardar_parquet(df, trabajo / f"barrido_{grupo}.parquet")
    pd.DataFrame(informe).to_csv(trabajo / f"informe_{grupo}.csv", index=False)
    print(f"Grupo {grupo}: {len(df):,} avisos únicos · {sum(x['paginas'] for x in informe):,} páginas · "
          f"{(time.time() - b.t0) / 60:.1f} min")
    return df


# ── 3. Juntar ────────────────────────────────────────────────────────────────
def juntar(subir: bool) -> dict:
    trabajo, salida = carpetas(RAIZ)
    df = pd.concat([pd.read_parquet(p) for p in sorted(trabajo.glob("barrido_*.parquet"))], ignore_index=True)
    informe = pd.concat([pd.read_csv(p) for p in sorted(trabajo.glob("informe_*.csv"))], ignore_index=True)
    filas = len(df)
    df = df.drop_duplicates("id_inmueble", keep="first").reset_index(drop=True)
    fecha = hoy()
    nacional = json.loads((trabajo / "nacional.json").read_text())["total_nacional"]
    reporte = {
        "fecha": fecha,
        "raiz": RAIZ,
        "segmentos": int(len(informe)),
        "paginas": int(informe["paginas"].sum()),
        "errores_pagina": int(informe["errores"].sum()),
        "filas_leidas": int(filas),
        "avisos_unicos": int(len(df)),
        "total_nacional_declarado": int(nacional),
        "cobertura": round(len(df) / nacional, 4),
        "segmentos_bajo_90pct": informe.loc[informe["unicos"] < 0.9 * informe["declarado"], "unidad"].tolist(),
        "escalones_sin_resolver": int(informe["sin_resolver"].sum()),
    }
    destino = salida / fecha / f"{RAIZ}.parquet"
    destino.parent.mkdir(parents=True, exist_ok=True)
    from src.almacen import COMPRESION
    guardar_parquet(df, destino, compression=COMPRESION)
    (destino.parent / "reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2))
    print(json.dumps(reporte, ensure_ascii=False, indent=2))
    print(f"💾 {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
    if subir:
        from src import almacen
        almacen.subir_archivo(destino, f"raw/{RAIZ}_http/fecha={fecha}/scrape.parquet")
        almacen.subir_archivo(destino.parent / "reporte.json", f"raw/{RAIZ}_http/fecha={fecha}/reporte.json")
    return reporte


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Barrido diario de Metrocuadrado (vivienda en venta, todo el país)")
    p.add_argument("--pausa", type=float, default=1.0, help="segundos entre páginas (cortesía)")
    sub = p.add_subparsers(dest="accion", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--grupos", type=int, default=4)
    b = sub.add_parser("barrer")
    b.add_argument("--grupo", type=int, required=True)
    b.add_argument("--solo", type=int, help="solo las N unidades más chicas del grupo (pruebas)")
    j = sub.add_parser("juntar")
    j.add_argument("--subir", action="store_true")
    args = p.parse_args()

    if args.accion == "plan":
        plan(args.grupos, args.pausa)
    elif args.accion == "barrer":
        barrer(args.grupo, args.pausa, args.solo)
    else:
        juntar(args.subir)
