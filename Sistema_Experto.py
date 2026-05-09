"""
Sistema_Experto.py — Motor de encadenamiento hacia adelante (Forward Chaining).
Reglas derivadas de NOM-172-SEMARNAT-2019, Tablas 5, 10 y 12.
"""

# ── Base de conocimiento (NOM-172, Tabla 5) ─────────────────────────────────
PM25_BANDS = [
    ("Buena",               "Bajo",                 (None, 25)),
    ("Aceptable",           "Moderado",             (25,   45)),
    ("Mala",                "Alto",                 (45,   79)),
    ("Muy Mala",            "Muy Alto",             (79,  147)),
    ("Extremadamente Mala", "Extremadamente Alto",  (147, None)),
]

RECOMENDACIONES = {
    "Buena": {
        "general":   "Disfruta las actividades al aire libre.",
        "sensibles": "Disfruta las actividades al aire libre.",
    },
    "Aceptable": {
        "general":   "Disfruta las actividades al aire libre.",
        "sensibles": "Considera reducir las actividades físicas vigorosas al aire libre.",
    },
    "Mala": {
        "general":   "Reduce las actividades físicas vigorosas al aire libre.",
        "sensibles": "Evita las actividades físicas (tanto moderadas como vigorosas) al aire libre",
    },
    "Muy Mala": {
        "general":   "Evita las actividades físicas moderadas y vigorosas al aire libre.",
        "sensibles": "No realices actividades al aire libre. Acude al médico si presentas síntomas.",
    },
    "Extremadamente Mala": {
        "general":   "Permanece en espacios interiores. Acude al médico si presentas síntomas respiratorios o cardiacos.",
        "sensibles": "Permanece en espacios interiores. Acude al médico si presentas síntomas respiratorios o cardiacos.",
    },
}

COLORES = {
    "Buena":               "Verde",
    "Aceptable":           "Amarillo",
    "Mala":                "Naranja",
    "Muy Mala":            "Rojo",
    "Extremadamente Mala": "Morado",
}


# ── Motor de inferencia ──────────────────────────────────────────────────────
def clasificar(pm25_value: float) -> dict:
    """Regla 1: clasificación de banda (NOM-172, Tabla 5)."""
    for calidad, riesgo, (lo, hi) in PM25_BANDS:
        if (lo is None or pm25_value > lo) and (hi is None or pm25_value <= hi):
            return {"calidad": calidad, "riesgo": riesgo}
    return {"calidad": "Extremadamente Mala", "riesgo": "Extremadamente Alto"}


def evaluar_tendencia(predicciones: list) -> str:
    """Regla 2: tendencia ascendente → posible contingencia."""
    if len(predicciones) < 2:
        return "estable"
    delta = predicciones[-1] - predicciones[0]
    if delta > 15:
        return "ascendente"
    elif delta < -15:
        return "descendente"
    return "estable"


def protocolo_contingencia(calidad: str, tendencia: str) -> str:
    """Regla 3: activación de fase de contingencia ambiental."""
    niveles_altos = {"Mala", "Muy Mala", "Extremadamente Mala"}
    if calidad in niveles_altos and tendencia == "ascendente":
        return "⚠️  ACTIVAR CONTINGENCIA AMBIENTAL — restringir fuentes de emisión."
    if calidad == "Extremadamente Mala":
        return "⚠️  CONTINGENCIA ACTIVA — medidas extremas en vigor."
    return "✅  Sin contingencia requerida."


def razonar(pm25_predicciones: list) -> dict:
    """
    Encadenamiento hacia adelante completo:
      Datos → Clasificación → Riesgo → Recomendaciones → Contingencia
    Recibe lista de 6 valores (uno por hora predicha).
    """
    # Clasificar basado en el peor valor predicho en las próximas 6h
    peor_valor = max(pm25_predicciones)
    clase      = clasificar(peor_valor)
    calidad    = clase["calidad"]
    tendencia  = evaluar_tendencia(pm25_predicciones)
    contingencia = protocolo_contingencia(calidad, tendencia)

    # Clasificar cada hora individualmente
    horas_detalle = []
    for i, val in enumerate(pm25_predicciones, start=1):
        c = clasificar(val)
        horas_detalle.append(
            f"  h+{i}: {val:6.1f} µg/m³  →  {c['calidad']} ({c['riesgo']})"
        )

    cadena = (
        f"Peor valor 6h: {peor_valor:.1f} µg/m³ "
        f"→ Calidad: {calidad} "
        f"→ Riesgo: {clase['riesgo']} "
        f"→ Tendencia: {tendencia} "
        f"→ {contingencia}"
    )

    return {
        "pm25_6h":           pm25_predicciones,          # lista completa
        "pm25_peor":         peor_valor,                  # hora más crítica
        "horas_detalle":     horas_detalle,               # texto por hora
        "calidad":           calidad,
        "riesgo":            clase["riesgo"],
        "color":             COLORES[calidad],
        "tendencia":         tendencia,
        "contingencia":      contingencia,
        "rec_general":       RECOMENDACIONES[calidad]["general"],
        "rec_sensibles":     RECOMENDACIONES[calidad]["sensibles"],
        "cadena_razonamiento": cadena,
    }
