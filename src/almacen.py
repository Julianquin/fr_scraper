"""
Almacenamiento de los datos en Cloudflare R2 (compatible con S3), en vez de en git.

El código sigue leyendo y escribiendo los parquet en data/ como siempre; este módulo
solo los sincroniza con el bucket:

  - `bajar`     trae los datos actuales del bucket a data/ (al empezar el CI o al trabajar en local)
  - `subir`     sube los datos actuales y el respaldo del día (al terminar el CI)
  - `raw`       sube lo scrapeado en el día (data/raw/<fecha>/*.csv) como un parquet
  - `historial` sube las fotos diarias del master guardadas en git (una sola vez, al migrar)
  - `retencion` pasa los respaldos del formato viejo a respaldos/historico/ (una sola vez)
  - `listar`    muestra qué hay en el bucket y cuánto pesa

Estructura del bucket (detalle en DATOS.md):
    master/listings.parquet                       foto actual (1 fila por aviso)
    master/events.parquet                         bitácora de eventos
    app/housing_clean.parquet                     dataset activo (mapa y alerta)
    raw/venta/fecha=AAAA-MM-DD/scrape.parquet     lo scrapeado ese día; nunca se borra
    respaldos/diario/fecha=AAAA-MM-DD/<archivo>   master y bitácora del día; R2 los borra a los 30 días
    respaldos/semanal/fecha=AAAA-MM-DD/<archivo>  los de cada lunes; se guardan siempre
    respaldos/historico/fecha=AAAA-MM-DD/...      las fotos que venían de git (ago–oct 2026)

La regla de 30 días es de R2 (bucket → Settings → Object lifecycle rules, prefijo
respaldos/diario/), no de este código.

Credenciales por variables de entorno (secrets en GitHub; .env en local, nunca en el repo):
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET

Uso:
    python -m src.almacen bajar
    python -m src.almacen subir
    python -m src.almacen raw --fecha 2026-10-24
    python -m src.almacen raw --fecha 2025-07-07 --carpeta data/raw   # CSV sueltos de jul-2025
    python -m src.almacen historial
    python -m src.almacen retencion
    python -m src.almacen listar
"""
from __future__ import annotations
import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, DATA_RAW

# clave en el bucket → archivo local
ACTUALES = {
    "master/listings.parquet": BASE_DIR / "data" / "master" / "listings.parquet",
    "master/events.parquet": BASE_DIR / "data" / "master" / "events.parquet",
    "app/housing_clean.parquet": BASE_DIR / "data" / "app" / "housing_clean.parquet",
}
RESPALDADOS = ["master/listings.parquet", "master/events.parquet"]
COMPRESION = "zstd"      # ~30–40% menos que snappy (el defecto); la usan todos los parquet


def _cargar_env() -> None:
    """Lee BASE_DIR/.env (KEY=valor por línea) sin pisar variables ya definidas."""
    env = BASE_DIR / ".env"
    if not env.exists():
        return
    for linea in env.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            os.environ.setdefault(clave.strip(), valor.strip().strip("'\""))


def configurado() -> bool:
    """Hay credenciales de R2 (en el entorno o en .env)."""
    _cargar_env()
    return all(os.environ.get(v) for v in
               ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"))


def _cliente():
    import boto3
    _cargar_env()
    faltan = [v for v in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
              if not os.environ.get(v)]
    if faltan:
        sys.exit(f"❌ Faltan variables de R2: {', '.join(faltan)} (secrets en GitHub o .env en local)")
    s3 = boto3.client(
        "s3", region_name="auto",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    return s3, os.environ["R2_BUCKET"]


def _existe(s3, bucket: str, clave: str) -> bool:
    from botocore.exceptions import ClientError
    try:
        s3.head_object(Bucket=bucket, Key=clave)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def bajar(claves: list[str] | None = None) -> None:
    """Trae del bucket los datos actuales. Si una clave no existe, deja el archivo local como está."""
    s3, bucket = _cliente()
    for clave in claves or ACTUALES:
        destino = ACTUALES[clave]
        if not _existe(s3, bucket, clave):
            print(f"⚠️  {clave} no está en el bucket; se conserva el archivo local")
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, clave, str(destino))
        print(f"⬇️  {clave} → {destino.relative_to(BASE_DIR)} ({destino.stat().st_size / 1e6:.1f} MB)")


def subir(fecha: str, claves: list[str] | None = None, respaldo: bool = True) -> None:
    """Sube los datos actuales y, con `respaldo`, una copia del master y la bitácora en
    respaldos/diario/ (y también en respaldos/semanal/ si `fecha` es lunes)."""
    s3, bucket = _cliente()
    carpetas = ["diario"] + (["semanal"] if pd.Timestamp(fecha).dayofweek == 0 else [])
    for clave in claves or ACTUALES:
        origen = ACTUALES[clave]
        if not origen.exists():
            print(f"⚠️  No existe {origen.relative_to(BASE_DIR)}; no se sube {clave}")
            continue
        s3.upload_file(str(origen), bucket, clave)
        print(f"⬆️  {origen.relative_to(BASE_DIR)} → {clave} ({origen.stat().st_size / 1e6:.1f} MB)")
        if respaldo and clave in RESPALDADOS:
            for carpeta in carpetas:
                copia = f"respaldos/{carpeta}/fecha={fecha}/{Path(clave).name}"
                s3.upload_file(str(origen), bucket, copia)
                print(f"   ↳ respaldo {copia}")


def subir_raw(fecha: str, carpeta: Path | None = None) -> None:
    """Junta los CSV scrapeados en data/raw/<fecha>/ en un parquet y lo sube.

    Con `carpeta`, toma solo los CSV sueltos de esa carpeta (sin subcarpetas): así se
    suben los scrapeos viejos que no están en una carpeta con fecha (los de julio de 2025).
    """
    if carpeta:
        archivos = sorted(Path(carpeta).glob("*.csv"))
    else:
        archivos = sorted((Path(DATA_RAW) / fecha).rglob("*.csv"))
    if not archivos:
        print(f"⚠️  No hay CSV en {carpeta or f'data/raw/{fecha}'}; nada que subir")
        return
    df = pd.concat([pd.read_csv(f, dtype=str).assign(archivo_origen=f.name) for f in archivos],
                   ignore_index=True)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression=COMPRESION)
    s3, bucket = _cliente()
    clave = f"raw/venta/fecha={fecha}/scrape.parquet"
    s3.put_object(Bucket=bucket, Key=clave, Body=buf.getvalue())
    print(f"⬆️  {len(archivos)} CSV ({len(df):,} filas) → {clave} ({buf.tell() / 1e6:.1f} MB)")


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(BASE_DIR), *args], check=True, capture_output=True).stdout


def subir_historial() -> None:
    """Sube a respaldos/historico/ las fotos del master y la bitácora guardadas en git (la
    última de cada día). Las que ya están en el bucket se saltan: se puede repetir sin costo."""
    s3, bucket = _cliente()
    for clave in RESPALDADOS:
        ruta = str(ACTUALES[clave].relative_to(BASE_DIR))
        por_dia = {}
        for linea in _git("log", "--reverse", "--format=%H %cs", "--", ruta).decode().splitlines():
            sha, dia = linea.split()
            por_dia[dia] = sha                          # la última foto de cada día gana
        subidas = 0
        for dia, sha in por_dia.items():
            copia = f"respaldos/historico/fecha={dia}/{Path(clave).name}"
            if _existe(s3, bucket, copia):
                continue
            try:
                contenido = _git("show", f"{sha}:{ruta}")
            except subprocess.CalledProcessError:
                continue                                # commit que borró el archivo
            s3.put_object(Bucket=bucket, Key=copia, Body=contenido)
            subidas += 1
        print(f"⬆️  {clave}: {subidas} fotos nuevas en respaldos/historico/ ({len(por_dia)} días en git)")


def migrar_retencion(aplicar: bool = False) -> None:
    """Pasa los respaldos del formato viejo (respaldos/fecha=…/) a respaldos/historico/fecha=…/.

    Sin `aplicar` solo muestra qué movería. Copia primero y borra el original solo si la
    copia quedó con el mismo tamaño. Se puede repetir: lo ya movido no aparece de nuevo.
    """
    s3, bucket = _cliente()
    viejos = [o for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="respaldos/fecha=")
              for o in pag.get("Contents", [])]
    if not viejos:
        print("✅ No quedan respaldos en el formato viejo")
        return
    total = sum(o["Size"] for o in viejos)
    print(f"{len(viejos)} respaldos en formato viejo ({total / 1e6:.0f} MB) → respaldos/historico/")
    if not aplicar:
        for o in viejos[:3]:
            print(f"   {o['Key']} → {o['Key'].replace('respaldos/', 'respaldos/historico/', 1)}")
        print("   … (simulación: nada se movió; agrega --aplicar para hacerlo)")
        return
    for o in viejos:
        nueva = o["Key"].replace("respaldos/", "respaldos/historico/", 1)
        s3.copy_object(Bucket=bucket, Key=nueva, CopySource={"Bucket": bucket, "Key": o["Key"]})
        if s3.head_object(Bucket=bucket, Key=nueva)["ContentLength"] != o["Size"]:
            sys.exit(f"❌ La copia de {o['Key']} no coincide; se detiene sin borrar el original")
        s3.delete_object(Bucket=bucket, Key=o["Key"])
    print(f"✅ {len(viejos)} respaldos movidos a respaldos/historico/")


def listar() -> None:
    s3, bucket = _cliente()
    total, n, por_prefijo = 0, 0, {}
    for pagina in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for o in pagina.get("Contents", []):
            total += o["Size"]
            n += 1
            partes = o["Key"].split("/")
            # respaldos se desglosa en diario/semanal/historico; el resto por carpeta raíz
            prefijo = "/".join(partes[:2]) if partes[0] == "respaldos" and not partes[1].startswith("fecha=") \
                else partes[0]
            por_prefijo[prefijo] = por_prefijo.get(prefijo, 0) + o["Size"]
    print(f"🪣 {bucket}: {n:,} objetos · {total / 1e9:.2f} GB (plan gratuito: 10 GB)")
    for p, s in sorted(por_prefijo.items()):
        print(f"   {p + '/':<22} {s / 1e6:>9.1f} MB")


if __name__ == "__main__":
    hoy = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="Sincroniza los datos con Cloudflare R2")
    sub = p.add_subparsers(dest="accion", required=True)
    b = sub.add_parser("bajar", help="trae los datos actuales del bucket a data/")
    b.add_argument("--solo", nargs="+", choices=list(ACTUALES), help="solo estas claves")
    s = sub.add_parser("subir", help="sube los datos actuales + respaldo del día")
    s.add_argument("--solo", nargs="+", choices=list(ACTUALES), help="solo estas claves")
    s.add_argument("--fecha", default=hoy)
    s.add_argument("--sin-respaldo", action="store_true")
    r = sub.add_parser("raw", help="sube lo scrapeado en data/raw/<fecha>/")
    r.add_argument("--fecha", default=hoy)
    r.add_argument("--carpeta", type=Path, help="CSV sueltos de esta carpeta (p. ej. data/raw para julio de 2025)")
    sub.add_parser("historial", help="sube las fotos del master guardadas en git (migración)")
    t = sub.add_parser("retencion", help="pasa los respaldos viejos a respaldos/historico/ (una vez)")
    t.add_argument("--aplicar", action="store_true", help="sin esto solo simula")
    sub.add_parser("listar", help="qué hay en el bucket y cuánto pesa")
    a = p.parse_args()

    if a.accion == "bajar":
        bajar(a.solo)
    elif a.accion == "subir":
        subir(a.fecha, a.solo, respaldo=not a.sin_respaldo)
    elif a.accion == "raw":
        subir_raw(a.fecha, a.carpeta)
    elif a.accion == "historial":
        subir_historial()
    elif a.accion == "retencion":
        migrar_retencion(a.aplicar)
    else:
        listar()
