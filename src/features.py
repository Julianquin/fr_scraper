from src.config import DATA_PROC
import pandas as pd

df = pd.read_parquet(DATA_PROC / "housing_preprocessed.parquet")


def generar_urls_completas():
    # Lista de ciudades y departamentos (ampliada)
    ciudades_departamentos = [
        # Principales
        ("bogota", "bogota-dc"),
        ("medellin", "antioquia"),
        ("cali", "valle-del-cauca"),
        ("barranquilla", "atlantico"),
        ("cartagena", "bolivar"),
        ("pereira", "risaralda"),
        ("bucaramanga", "santander"),
        ("cucuta", "norte-de-santander"),
        ("ibague", "tolima"),
        ("villavicencio", "meta"),
        ("manizales", "caldas"),
        ("pasto", "narino"),
        ("monteria", "cordoba"),
        ("santa-marta", "magdalena"),
        ("armenia", "quindio"),
        
        # Municipios estratégicos
        ("soacha", "cundinamarca"),
        ("facatativa", "cundinamarca"),
        ("girardot", "cundinamarca"),
        ("zipaquira", "cundinamarca"),
        ("envigado", "antioquia"),
        ("sabaneta", "antioquia"),
        ("bello", "antioquia"),
        ("itagui", "antioquia"),
        ("rionegro", "antioquia"),
        ("palmira", "valle-del-cauca"),
        ("yumbo", "valle-del-cauca"),
        ("tulua", "valle-del-cauca"),
        ("soledad", "atlantico"),
        ("dosquebradas", "risaralda"),
        ("jamundi", "valle-del-cauca"),
        ("girardot", "cundinamarca"),
        ("cajica", "cundinamarca"),
        ("chia", "cundinamarca"),
        ("la-ceja", "antioquia"),
        ("el-retiro", "antioquia"),
        
        # Turísticas
        ("san-andres", "san-andres"),
        ("leticia", "amazonas"),
        ("melgar", "tolima"),
        ("santa-fe-de-antioquia", "antioquia")
    ]

    # Tipos de propiedades (individuales y combinados)
    tipos_propiedades = [
        "casas",
        "apartamentos",
        "fincas",
        "apartaestudios",
        "cabanas",
        "casas-campestres",
        "casas-lotes",
        "casas-y-apartamentos",
        "casas-y-fincas",
        "apartamentos-y-apartaestudios",
        "casas-campestres-y-cabanas",
        "casas-y-apartamentos-y-fincas-y-apartaestudios-y-cabanas-y-casas-campestres-y-casas-lotes"
    ]

    # URLs base
    urls = [
        # "https://www.fincaraiz.com.co/venta",
        # "https://www.fincaraiz.com.co/inmobiliarias",
        # "https://www.fincaraiz.com.co/constructoras"
    ]

    # Generar URLs por tipo de propiedad y ubicación
    for ciudad, departamento in ciudades_departamentos:
        # URLs generales por ciudad
        urls.append(f"https://www.fincaraiz.com.co/venta/{ciudad}/{departamento}")
        urls.append(f"https://www.fincaraiz.com.co/proyectos-vivienda/{ciudad}/{departamento}")
        
        # URLs por tipo de propiedad específico
        for tipo in tipos_propiedades:
            urls.append(f"https://www.fincaraiz.com.co/venta/{tipo}/{ciudad}/{departamento}")
        
        # URLs especiales para proyectos de vivienda
        urls.append(f"https://www.fincaraiz.com.co/proyectos-vivienda-nueva/{ciudad}/{departamento}")

    # Eliminar duplicados y mantener orden
    return list(dict.fromkeys(urls))
