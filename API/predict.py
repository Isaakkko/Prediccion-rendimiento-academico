import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

model_regresion = load_model('model1_featureengineering.keras')
with open('scaler_regresion.pkl', 'rb') as f:
    scaler_regresion = pickle.load(f)

model_riesgo = load_model('modelo_riesgo.keras')
with open('scaler.pkl', 'rb') as f:
    scaler_riesgo = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

columnas_regresion = ['age','Medu','Fedu','traveltime','studytime','failures','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','avg_grade','parent_edu_avg','alcohol_total','absence_rate','school_MS','sex_M','address_U','famsize_LE3','Pstatus_T','Mjob_health','Mjob_other','Mjob_services','Mjob_teacher','Fjob_health','Fjob_other','Fjob_services','Fjob_teacher','reason_home','reason_other','reason_reputation','guardian_mother','guardian_other','schoolsup_yes','famsup_yes','paid_yes','activities_yes','nursery_yes','higher_yes','internet_yes','romantic_yes','subject_portuguese']
columnas_riesgo = ['G1','G2','failures','studytime','absences']

def predecir_grade(datos: dict):
    df_input = pd.DataFrame([datos])
    df_input['avg_grade'] = (df_input['G1'] + df_input['G2']) / 2
    df_input['parent_edu_avg'] = (df_input['Medu'] + df_input['Fedu']) / 2
    df_input['alcohol_total'] = df_input['Dalc'] + df_input['Walc']
    df_input['absence_rate'] = df_input['absences'] / df_input['absences'].max()
    df_input = pd.get_dummies(df_input, drop_first=True)
    bool_cols = df_input.select_dtypes(include='bool').columns
    df_input[bool_cols] = df_input[bool_cols].astype(int)
    df_input = df_input.reindex(columns=columnas_regresion, fill_value=0)
    X_scaled = scaler_regresion.transform(df_input)
    pred = model_regresion.predict(X_scaled)
    return round(float(pred[0][0]), 2)

def predecir_riesgo(datos: dict):
    df_input = pd.DataFrame([datos])
    X_scaled = scaler_riesgo.transform(df_input)
    prob = model_riesgo.predict(X_scaled)
    clase = le.inverse_transform(np.argmax(prob, axis=1))[0]
    return {
        "prediccion": clase,
        "probabilidades": dict(zip(le.classes_, prob[0].round(4).tolist()))
    }