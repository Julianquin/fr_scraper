"""
Las aprobaciones del canal, una por archivo: canal/aprobaciones/<mensaje>.json en R2 (data/canal/aprobaciones/ sin
claves). Solo depende de boto3, para que anotarla sea rápido.

Por qué (19-nov): cada «Publicar» lanza una ejecución de canal.yml, y las que escriben canal/estado.json van de una
en una (grupo de concurrencia «canal»). Pero GitHub deja una sola en espera por grupo y cancela la anterior: tres
«Publicar» en 8 s → la del medio (Ciudad Jardín) se canceló y su aprobación se perdió. Ahora la aprobación se anota
primero aquí, en un trabajo sin grupo (nada que se pise: un archivo por mensaje), y el paso que escribe el estado
recoge TODAS las anotadas (src/canal.py: aprobar y publicar). Si GitHub cancela una ejecución en espera, la
siguiente, o la publicación de las 07:00, recoge su aprobación.

Uso (canal.yml, trabajo «marcar»):
    python -m src.aprobaciones marcar --mensaje 1234
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PREFIJO = "canal/aprobaciones/"
LOCAL = Path(__file__).resolve().parent.parent / "data" / "canal" / "aprobaciones"
CLAVES = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


def _r2():
    """(cliente, bucket), o None sin claves (en local)."""
    if not all(os.environ.get(v) for v in CLAVES):
        return None
    import boto3
    s3 = boto3.client("s3", region_name="auto",
                      endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                      aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])
    return s3, os.environ["R2_BUCKET"]


def marcar(mensaje: int) -> None:
    """Anota que se pulsó «Publicar» en la propuesta `mensaje`."""
    cuerpo = json.dumps({"mensaje": int(mensaje), "cuando": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    if (r2 := _r2()) is None:
        LOCAL.mkdir(parents=True, exist_ok=True)
        (LOCAL / f"{int(mensaje)}.json").write_text(cuerpo)
    else:
        s3, bucket = r2
        s3.put_object(Bucket=bucket, Key=f"{PREFIJO}{int(mensaje)}.json", Body=cuerpo.encode(),
                      ContentType="application/json")
    print(f"✅ Aprobación anotada ({mensaje})")


def pendientes() -> list[int]:
    """Los mensajes con «Publicar» pulsado que aún no se han recogido."""
    if (r2 := _r2()) is None:
        return sorted(int(p.stem) for p in LOCAL.glob("*.json")) if LOCAL.exists() else []
    s3, bucket = r2
    claves = [o["Key"] for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=PREFIJO)
              for o in pag.get("Contents", [])]
    return sorted(int(Path(k).stem) for k in claves if Path(k).stem.isdigit())


def borrar(mensajes: list[int]) -> None:
    """Las anotaciones ya recogidas (en la cola o publicadas)."""
    if not mensajes:
        return
    if (r2 := _r2()) is None:
        for m in mensajes:
            (LOCAL / f"{int(m)}.json").unlink(missing_ok=True)
        return
    s3, bucket = r2
    s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": f"{PREFIJO}{int(m)}.json"} for m in mensajes]})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("accion", choices=["marcar", "pendientes"])
    p.add_argument("--mensaje", type=int)
    a = p.parse_args()
    if a.accion == "marcar":
        if not a.mensaje:
            raise SystemExit("❌ Falta --mensaje")
        marcar(a.mensaje)
    else:
        print(pendientes())


if __name__ == "__main__":
    main()
