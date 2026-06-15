# Streamlit - Predicción de Rendimiento Académico

## Ejecución

Desde la raíz del repositorio:

```bash
pip install -r requirements.txt
streamlit run APP/Home.py
```

## Qué incluye

- Dashboard profesional con KPIs.
- Sistema de alertas tempranas por nivel de riesgo.
- Predicción individual de rendimiento académico.
- Recomendaciones automáticas de intervención.
- Reporte descargable en CSV.
- Lectura automática de datos desde `DATA/PROCESSED`, `DATA/RAW` o dataset demo si no hay CSV/XLSX.
- Preparado para integrar modelos Keras guardados en `MODELS`.

## Variables esperadas

La app funciona mejor si el dataset contiene columnas similares a:

- `id_estudiante`
- `carrera`
- `nivel`
- `asistencia`
- `promedio_actual`
- `entrega_tareas`
- `participacion`
- `materias_reprobadas`
- `ausencias`
- `apoyo_familiar`
- `horas_estudio_semana`

También reconoce algunos nombres en inglés como `attendance`, `gpa`, `study_hours`, `failures`, etc.
