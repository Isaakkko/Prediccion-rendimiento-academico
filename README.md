# Predicción de Rendimiento Académico

Proyecto de análisis de datos e inteligencia artificial orientado a la predicción del rendimiento académico y la identificación temprana de estudiantes en riesgo.

**Colegio Universitario de Cartago — Costa Rica**

## Integrantes

- Isaac Ulloa Calvo
- Jeffrey Jiménez Cordero
- Felipe Montenegro Artavia

## Descripción

El proyecto analiza información académica, familiar y de comportamiento de estudiantes para identificar patrones relacionados con su desempeño.

A partir de los datos se desarrollaron modelos de redes neuronales artificiales capaces de:

- Estimar la calificación final del estudiante (`G3`) mediante un modelo de regresión.
- Clasificar el nivel de riesgo académico mediante un modelo multiclase.
- Analizar los factores que presentan mayor relación con el rendimiento.
- Apoyar la detección temprana de posibles dificultades académicas.

El sistema integra análisis exploratorio de datos, preprocesamiento, ingeniería de características, entrenamiento de modelos, una API REST desarrollada con FastAPI y una aplicación web construida con Streamlit.

## Fuente de datos

El proyecto utiliza datos de rendimiento estudiantil correspondientes a las asignaturas de Matemáticas y Portugués.

Los datos contienen variables académicas, demográficas, familiares y de hábitos de estudio. Entre las variables analizadas se encuentran:

- `G1`: calificación del primer periodo.
- `G2`: calificación del segundo periodo.
- `G3`: calificación final.
- `studytime`: tiempo de estudio.
- `failures`: cantidad de asignaturas reprobadas.
- `absences`: ausencias.
- `Medu` y `Fedu`: nivel educativo de los padres.
- `Dalc` y `Walc`: consumo de alcohol entre semana y fines de semana.
- `higher`: intención de continuar estudios superiores.
- `famrel`: calidad de las relaciones familiares.

Los archivos originales se almacenan en `DATA/RAW` y los datos preparados para análisis y modelado se encuentran en `DATA/PROCESSED`.

## Análisis exploratorio de datos

El análisis exploratorio se desarrolla en `NOTEBOOKS/01_EDA.ipynb`.

El EDA incluye:

- Revisión de estructura y tipos de datos.
- Análisis de valores faltantes.
- Distribución de variables.
- Análisis de calificaciones.
- Matrices de correlación.
- Comparación de variables académicas y de comportamiento.
- Identificación de factores relacionados con el rendimiento académico.

Las calificaciones previas `G1` y `G2` representan variables especialmente relevantes para la estimación de la calificación final.

## Ingeniería de características

Para ampliar la información disponible para el modelo de regresión se crearon nuevas variables:

- `avg_grade`: promedio de `G1` y `G2`.
- `parent_edu_avg`: promedio del nivel educativo de la madre y el padre.
- `alcohol_total`: combinación de `Dalc` y `Walc`.
- `absence_rate`: proporción normalizada de ausencias.

Estas variables permiten representar de forma más compacta patrones académicos, familiares y de comportamiento.

## Modelos predictivos

### Modelo de regresión

El modelo de regresión tiene como objetivo estimar la calificación final `G3`.

El desarrollo y evaluación se documentan principalmente en:

- `NOTEBOOKS/03_ANN_Modelo1.ipynb`
- `NOTEBOOKS/03_ANN_Modelo1_FeatureEngineering.ipynb`
- `NOTEBOOKS/05_Comparacion_Modelos.ipynb`

El modelo utiliza TensorFlow y Keras para construir una red neuronal artificial.

### Modelo de clasificación multiclase

El segundo modelo clasifica el nivel de riesgo académico del estudiante.

Su desarrollo se encuentra en:

- `NOTEBOOKS/Modelo_Multiclase2.ipynb`
- `NOTEBOOKS/Modelo_Multiclase_Engineering.ipynb`

Para la inferencia de riesgo se utilizan variables académicas como:

- `G1`
- `G2`
- `failures`
- `studytime`
- `absences`

Los modelos y objetos de preprocesamiento entrenados se almacenan en `MODELS`.

## API REST

La carpeta `API` contiene una API desarrollada con FastAPI para exponer los modelos predictivos mediante solicitudes HTTP.

### Endpoint raíz

`GET /`

Permite verificar que la API se encuentra disponible.

### Predicción de calificación final

`POST /predict/grade`

Recibe las características de un estudiante y retorna la estimación de `G3`.

Ejemplo de respuesta:

```json
{
  "G3_predicho": 13.42
}
```

### Predicción del nivel de riesgo

`POST /predict/risk_level`

Recibe las variables principales del modelo multiclase.

Ejemplo de entrada:

```json
{
  "G1": 10,
  "G2": 9,
  "failures": 1,
  "studytime": 2,
  "absences": 6
}
```

La respuesta contiene la clase predicha y las probabilidades calculadas para cada nivel.

## Aplicación web

La carpeta `APP` contiene una aplicación desarrollada con Streamlit.

La interfaz incluye:

- Dashboard de indicadores académicos.
- Visualización de estudiantes por nivel de riesgo.
- Alertas tempranas.
- Predicción individual.
- Recomendaciones de intervención.
- Exportación de reportes en formato CSV.
- Lectura de archivos desde `DATA/PROCESSED` y `DATA/RAW`.

La aplicación también dispone de lógica de respaldo basada en reglas cuando el servicio de predicción no se encuentra disponible.

## Estructura del proyecto

```text
Prediccion-rendimiento-academico/
│
├── API/
│   ├── main.py
│   ├── predict.py
│   └── schemas.py
│
├── APP/
│   ├── .streamlit/
│   ├── Home.py
│   ├── README_API_STREAMLIT.md
│   ├── README_STREAMLIT.md
│   └── requirements.txt
│
├── DATA/
│   ├── RAW/
│   ├── PROCESSED/
│   └── DATA.ipynb
│
├── MODELS/
│   ├── model1.h5
│   ├── model1_featureengineering.h5
│   ├── modelo_riesgo.keras
│   ├── modelo_riesgo_engineering.keras
│   ├── scaler.pkl
│   ├── scaler_fe.pkl
│   ├── label_encoder.pkl
│   └── label_encoder_fe.pkl
│
├── NOTEBOOKS/
│   ├── 01_EDA.ipynb
│   ├── 03_ANN_Modelo1.ipynb
│   ├── 03_ANN_Modelo1_FeatureEngineering.ipynb
│   ├── 05_Comparacion_Modelos.ipynb
│   ├── Modelo_Multiclase2.ipynb
│   └── Modelo_Multiclase_Engineering.ipynb
│
├── SRC/
│   └── TRAIN/
│       ├── config.py
│       └── data_prep.py
│
├── README.md
└── REQUIREMENTS.txt
```

## Tecnologías utilizadas

- Python
- TensorFlow
- Keras
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Plotly
- FastAPI
- Uvicorn
- Streamlit
- Jupyter Notebook

## Instalación

Se recomienda utilizar Python 3.11.

Clonar el repositorio:

```bash
git clone https://github.com/Isaakkko/Prediccion-rendimiento-academico.git
cd Prediccion-rendimiento-academico
```

Crear el entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar las dependencias:

```bash
python -m pip install -r REQUIREMENTS.txt
```

## Ejecución de la aplicación Streamlit

Desde la raíz del repositorio:

```bash
python -m streamlit run APP/Home.py
```

La aplicación estará disponible normalmente en:

`http://localhost:8501`

## Ejecución de la API

Antes de ejecutar la API, los archivos de modelo y preprocesamiento requeridos por `API/main.py` deben estar disponibles en las rutas configuradas.

Desde la raíz del repositorio:

```bash
python -m uvicorn API.main:app --reload
```

La documentación interactiva de FastAPI estará disponible normalmente en:

`http://127.0.0.1:8000/docs`

## Objetivo del proyecto

El objetivo del proyecto es demostrar la aplicación de técnicas de inteligencia artificial y redes neuronales artificiales en el análisis del rendimiento académico.

La solución busca apoyar la identificación temprana de estudiantes con posibles dificultades, facilitando el análisis de factores de riesgo y la generación de información útil para estrategias de acompañamiento académico.

## Uso académico

Proyecto desarrollado con fines académicos. Las predicciones generadas por los modelos deben interpretarse como una herramienta de apoyo al análisis y no como un criterio único para la toma de decisiones sobre un estudiante.
