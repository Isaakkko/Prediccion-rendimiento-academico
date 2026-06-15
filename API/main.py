#Librerias
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

#Inicializar la aplicacion FatAPI
app = FastAPI()

#Cargar el modelo de regresion y su scaler
model_regresion = load_model("model1_feature.keras")
with open("scaler_regresion.pkl", "rb") as f2:
    scaler_regresion = pickle.load(f2)


#Cargar el modelo de clasificacion multiclase, su scaler y label encoder
model_riesgo = load_model("modelo_riesgo.keras")
with open("scaler.pkl", "rb") as f2:
    scaler_riesgo = pickle.load(f2)
with open("label_encoder.pkl", "rb") as f2:
    le = pickle.load(f2)

with open("columnas_regresion.pkl", "rb") as f2:
    columnas_regresion = pickle.load(f2)


#Columnas esperadas para cada modelo
columnas_regresion = ["age","Medu","Fedu","traveltime","studytime","failures","famrel","freetime","goout","Dalc","Walc","health","absences","G1","G2","avg_grade","parent_edu_avg","alcohol_total","absence_rate","school_MS","sex_M","address_U","famsize_LE3","Pstatus_T","Mjob_health","Mjob_other","Mjob_services","Mjob_teacher","Fjob_health","Fjob_other","Fjob_services","Fjob_teacher","reason_home","reason_other","reason_reputation","guardian_mother","guardian_other","schoolsup_yes","famsup_yes","paid_yes","activities_yes","nursery_yes","higher_yes","internet_yes","romantic_yes","subject_portuguese"]
columnas_riesgo = ["G1","G2","failures","studytime","absences"]

#Schema de entrada para la regresion
class EstudianteGrade(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    G1: int
    G2: int
    subject: str

#Esquema de entrada para el modelo multiclase
class EstudianteRiesgo(BaseModel):
    G1: int
    G2: int
    failures: int
    studytime: int
    absences: int

#Endpoint raiz
@app.get("/")
def root():
    return {"mensaje": "API de prediccion academica"}

#Endpoint para predecir la nota final G3
@app.post("/predict/grade")
def predict_grade(estudiante: EstudianteGrade):
    #Convertir input a dataframe
    df_input = pd.DataFrame([estudiante.dict()])
    #Feature engineering igual al del entrenamiento
    df_input["avg_grade"] = (df_input["G1"] + df_input["G2"]) / 2
    df_input["parent_edu_avg"] = (df_input["Medu"] + df_input["Fedu"]) / 2
    df_input["alcohol_total"] = df_input["Dalc"] + df_input["Walc"]
    df_input["absence_rate"] = df_input["absences"] / df_input["absences"].max()
    #Encodear las variables categoricas
    df_input = pd.get_dummies(df_input, drop_first=True)
    #Alinear las columnas con las del entrenamiento
    df_input = df_input.reindex(columns=columnas_regresion, fill_value=0)
    bool_cols = df_input.select_dtypes(include="bool").columns
    #Escalar y predecir
    df_input[bool_cols] = df_input[bool_cols].astype(int)
    X_scaled = scaler_regresion.transform(df_input)
    pred = model_regresion.predict(X_scaled)
    return {"G3_predicho": round(float(pred[0][0]), 2)}

#Endpoint para predecir el nivel de riesgo
@app.post("/predict/risk_level")
def predict_risk_level(estudiante: EstudianteRiesgo):
    #Convertir input a dataframe
    df_input = pd.DataFrame([estudiante.dict()])
    #Escalar y predecir
    X_scaled = scaler_riesgo.transform(df_input)
    prob = model_riesgo.predict(X_scaled)
    #Obtener la clase con mayor probabilidad
    clase = le.inverse_transform(np.argmax(prob, axis=1))[0]
    return {
        "prediccion": clase,
        "probabilidades": dict(zip(le.classes_, prob[0].round(4).tolist()))
    }
