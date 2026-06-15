# 🎓 Predicción de Rendimiento Académico
**Proyecto — Análisis y Modelo Predictivo**  
**Colegio Universitario de Cartago — Costa Rica**

**Integrantes:**
- Isaac Ulloa Calvo
- Jeffrey Jiménez Cordero
- Felipe Montenegro Artavia

---

## 📌 Descripción General

Este proyecto desarrolla un sistema de análisis y predicción del rendimiento académico de estudiantes.  
Incluye:

- Análisis exploratorio de datos (EDA)
- Visualizaciones interactivas mediante una aplicación web
- API REST para consumo del modelo predictivo
- Modelos supervisados de clasificación y redes neuronales para predecir el desempeño estudiantil

El objetivo es identificar los factores que influyen en el rendimiento académico y construir un modelo capaz de estimar la probabilidad de éxito o riesgo de un estudiante según sus características.

---

## 📁 Fuente de Datos

### 🗂 Dataset Principal
Datos históricos de estudiantes con variables académicas, socioeconómicas y de comportamiento.

Variables incluidas:
- Calificaciones previas
- Asistencia
- Factores socioeconómicos
- Variables de comportamiento y hábitos de estudio

---

## 🧪 Análisis Exploratorio y Visualización (EDA)

El análisis incluye:

### ✔ Distribución y frecuencia
- Distribución de calificaciones por grupo
- Frecuencia de variables categóricas

### ✔ Correlaciones y mapas de calor
- Variables académicas vs rendimiento
- Factores externos vs desempeño

### ✔ Resultados clave
- Identificación de variables con mayor impacto
- Comparación entre perfiles de estudiantes en riesgo y sin riesgo

---

## 🤖 Modelo Predictivo

### 🎯 Tipo de problema
**Clasificación** — Predecir el rendimiento académico del estudiante.

### 🔢 Algoritmos utilizados
- Redes Neuronales (TensorFlow / Keras)
- Modelos de clasificación con scikit-learn

### 🧩 Variables de entrada
- Calificaciones anteriores
- Asistencia
- Variables socioeconómicas
- Hábitos de estudio

### 🎯 Variable objetivo
- **Rendimiento** (Alto / Medio / Bajo o clasificación binaria)

---

## 🌐 API REST

El proyecto expone una API construida con **FastAPI** y **Uvicorn** que permite:

- Recibir datos de un estudiante vía JSON
- Retornar la predicción del modelo en tiempo real
- Integrarse con la aplicación web (APP)

---

## 📱 Aplicación Web

Se desarrolló una interfaz web interactiva con **Streamlit** que permite:

- Ingresar los datos del estudiante
- Visualizar la predicción de rendimiento
- Explorar los resultados del análisis EDA

---
---

## ⚙️ Requerimientos Técnicos

### 🐍 Python 3.10+

### 📚 Librerías principales:
- tensorflow / keras
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- streamlit
- fastapi
- uvicorn
- requests

---

## 📈 Resultados Esperados

- Identificación de las variables con mayor influencia en el rendimiento
- Perfil de estudiantes en riesgo académico
- Predicción del desempeño mediante redes neuronales
- API funcional para consumo del modelo
- Interfaz web para visualización e inferencia en tiempo real

---

## 🧾 Conclusiones

- Las variables académicas previas son los predictores más fuertes del rendimiento futuro.
- Los modelos de redes neuronales ofrecen mayor capacidad para capturar patrones complejos.
- La combinación de una API REST con una interfaz Streamlit facilita el acceso al modelo de forma práctica.
- El análisis exploratorio revela patrones claros que distinguen a estudiantes en riesgo.

---

# Fin
