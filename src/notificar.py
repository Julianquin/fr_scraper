"""
Envía un mensaje de texto (p. ej. la alerta de src/oportunidades.py) por Telegram y/o WhatsApp.

Usa los canales que tengan credenciales en variables de entorno (secrets en GitHub Actions):
  - Telegram:  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID (uno o varios, separados por coma;
               un canal privado con el bot como admin también es un chat id, -100…)
  - WhatsApp:  CALLMEBOT_PHONE + CALLMEBOT_APIKEY   (CallMeBot, solo uso personal)

El texto viene con formato de WhatsApp (*negrita*, _cursiva_); para Telegram se convierte
a HTML. Los mensajes largos se parten por bloques (una oportunidad por bloque).

Uso:
    python -m src.notificar reports/oportunidades/cali_2026-10-23_whatsapp.txt
"""
from __future__ import annotations
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

LIMITE = {"telegram": 3500, "whatsapp": 1500}   # caracteres por mensaje (con margen)


def partir(texto: str, limite: int) -> list[str]:
    """Agrupa los bloques (separados por línea en blanco) en mensajes de hasta `limite`."""
    partes, actual = [], ""
    for bloque in texto.strip().split("\n\n"):
        if actual and len(actual) + len(bloque) + 2 > limite:
            partes.append(actual)
            actual = bloque
        else:
            actual = f"{actual}\n\n{bloque}" if actual else bloque
    return partes + [actual] if actual else partes


def _a_html(texto: str) -> str:
    t = html.escape(texto)
    t = re.sub(r"\*(.+?)\*", r"<b>\1</b>", t)
    return re.sub(r"(?m)^_(.+)_$", r"<i>\1</i>", t)


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def telegram(texto: str, token: str, chat_id: str) -> None:
    for parte in partir(texto, LIMITE["telegram"]):
        datos = urllib.parse.urlencode({"chat_id": chat_id, "text": _a_html(parte),
                                        "parse_mode": "HTML",
                                        "disable_web_page_preview": "true"}).encode()
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage",
                                    data=datos, timeout=60) as r:
            if not json.load(r).get("ok"):
                raise RuntimeError("Telegram rechazó el mensaje")


def whatsapp(texto: str, phone: str, apikey: str) -> None:
    for parte in partir(texto, LIMITE["whatsapp"]):
        q = urllib.parse.urlencode({"phone": phone, "text": parte, "apikey": apikey})
        respuesta = _get(f"https://api.callmebot.com/whatsapp.php?{q}")
        if "error" in respuesta.lower():
            raise RuntimeError(f"CallMeBot: {respuesta[:200]}")
        time.sleep(3)   # CallMeBot no documenta límites: espaciamos los envíos


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Uso: python -m src.notificar <archivo.txt>")
    texto = open(sys.argv[1], encoding="utf-8").read()
    env = os.environ.get

    enviados = []
    if env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID"):
        # Varios destinos separados por coma: tu chat, el canal de agentes, etc.
        chats = [c.strip() for c in env("TELEGRAM_CHAT_ID").split(",") if c.strip()]
        for chat in chats:
            telegram(texto, env("TELEGRAM_BOT_TOKEN"), chat)
        enviados.append(f"Telegram ({len(chats)} destinos)")
    if env("CALLMEBOT_PHONE") and env("CALLMEBOT_APIKEY"):
        whatsapp(texto, env("CALLMEBOT_PHONE"), env("CALLMEBOT_APIKEY"))
        enviados.append("WhatsApp")
    if not enviados:
        sys.exit("❌ No hay credenciales: define TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
                 "o CALLMEBOT_PHONE/CALLMEBOT_APIKEY.")
    print(f"✅ Enviado por {' y '.join(enviados)}")


if __name__ == "__main__":
    main()
