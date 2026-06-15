# Conexión de Streamlit con API de modelos entrenados

Este Streamlit está preparado para consumir un endpoint HTTP de predicción.

## Ejecutar Streamlit

```powershell
python -m pip install -r requirements.txt
python -m streamlit run APP/Home.py
```

## Configurar API

En el menú lateral de Streamlit aparece:

```text
URL del endpoint predict
```

Valor recomendado por defecto:

```text
http://127.0.0.1:8000/predict
```

Si su API corre en Flask normalmente sería algo como:

```text
http://127.0.0.1:5000/predict
```

## JSON que Streamlit envía al API

```json
{
  "id_estudiante": "EST-NUEVO",
  "carrera": "Ingeniería",
  "nivel": "I",
  "asistencia": 78,
  "promedio_actual": 72,
  "entrega_tareas": 75,
  "participacion": 65,
  "materias_reprobadas": 1,
  "ausencias": 4,
  "apoyo_familiar": 3,
  "horas_estudio_semana": 8
}
```

## JSON recomendado que debe devolver el API

```json
{
  "rendimiento_predicho": 74.5,
  "puntaje_riesgo": 25.5,
  "riesgo_desercion": "Riesgo bajo",
  "alerta_temprana": "Sin alerta crítica",
  "recomendaciones": [
    "Mantener seguimiento preventivo mensual",
    "Reforzar hábitos de estudio actuales"
  ]
}
```

Los valores válidos para `riesgo_desercion` son:

| Valor | Significado |
|---|---|
| `"Riesgo alto"` | Intervención urgente |
| `"Riesgo medio"` | Seguimiento preventivo |
| `"Riesgo bajo"` | Sin alerta crítica |

También acepta respuestas con nombres alternativos como `prediction`, `prediccion`, `risk_score`, `risk_level`, `probability` o `recommendations`.

## Uso en dashboard completo

En el menú lateral active:

```text
Usar API para el dashboard completo
```

Cuando está activo, Streamlit manda cada estudiante del dataset al API. Si el API falla, la app usa reglas locales como respaldo para no detener la presentación.