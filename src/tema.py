"""
La presentación de los informes en PDF (src/informe_semanal.py, src/informe_captacion.py): el logo, los colores y la
fuente de la marca, si están en data/marca/ (de R2 `marca/`, que bajan los workflows antes de generar los informes);
si no, el estilo neutro de siempre. El repositorio es público: aquí no hay nada de la marca, solo dónde buscarla, y
nada de lo que se lee se imprime.

    data/marca/config.json       {"contacto", "nombre", "cuenta", "canal", "colores": {"principal", "texto2", "linea"}}
    data/marca/logo.svg          el logo horizontal (en vector)
    data/marca/fuente-{regular,semibold,bold}.ttf    la fuente de la marca (opcional)

Lo sube a R2 el repositorio privado de la marca.
"""
from __future__ import annotations

import base64
import json
from functools import lru_cache

from src.config import BASE_DIR

CARPETA = BASE_DIR / "data" / "marca"
PESOS = {"regular": 400, "semibold": 600, "bold": 700}


@lru_cache(maxsize=1)
def marca() -> dict:
    """Lo que haya de la marca en data/marca/ (vacío si no hay nada)."""
    c = {}
    if (CARPETA / "config.json").exists():
        c = json.loads((CARPETA / "config.json").read_text(encoding="utf-8"))
    logo = CARPETA / "logo.svg"
    c["logo"] = ("data:image/svg+xml;base64," + base64.b64encode(logo.read_bytes()).decode()) if logo.exists() else None
    c["fuentes"] = {p: (CARPETA / f"fuente-{p}.ttf").as_uri() for p in PESOS if (CARPETA / f"fuente-{p}.ttf").exists()}
    return c


def css() -> str:
    """Las reglas que se añaden al final del <style> de un informe: la fuente de la marca en títulos y cifras, su
    color principal (--blue) y la cabecera de cada página con el logo. Vacío sin marca."""
    m, reglas = marca(), []
    for peso, uri in m["fuentes"].items():
        reglas.append(f"@font-face {{ font-family: Marca; src: url('{uri}'); font-weight: {PESOS[peso]}; }}")
    if m["fuentes"]:
        reglas.append("h1, h2, h3, .kicker, .t-value { font-family: Marca, Inter, 'DejaVu Sans', sans-serif; }")
    col = m.get("colores") or {}
    variables = {"--blue": col.get("principal"), "--ink-2": col.get("texto2"), "--line": col.get("linea")}
    if any(variables.values()):
        reglas.append(":root { " + " ".join(f"{k}: {v};" for k, v in variables.items() if v) + " }")
        if col.get("principal"):                # también donde el informe escribe el color sin variable
            p = col["principal"]
            reglas.append(f"h1, h2 {{ color: {p}; }} a {{ color: {p}; }} .resumen {{ border-left-color: {p}; }}")
    if m["logo"]:
        reglas.append(".marca-cabecera { position: running(cabecera); }")
        reglas.append(".marca-cabecera img { height: 7mm; }")
        reglas.append("@page { margin-top: 19mm; @top-left { content: element(cabecera); vertical-align: bottom; "
                      "padding-bottom: 3mm; } }")
    return "\n  ".join(reglas)


def cabecera() -> str:
    """El elemento con el logo que se repite arriba de cada página (va al principio del <body>)."""
    m = marca()
    return f'<div class="marca-cabecera"><img src="{m["logo"]}" alt=""></div>' if m["logo"] else ""


def prefijo() -> str:
    """'Nombre · ' para el pie de página, o vacío."""
    nombre = marca().get("nombre")
    return f"{nombre} · " if nombre else ""


def contacto_pie() -> str:
    """'cuenta · canal' para el centro del pie de página, o vacío."""
    m = marca()
    return " · ".join(x for x in (m.get("cuenta"), m.get("canal")) if x)
