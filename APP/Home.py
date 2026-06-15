# Sistema de Predicción de Rendimiento Académico
# Dashboard Streamlit con alertas tempranas e intervenciones
# Ejecutar desde la raíz del repo:
#   streamlit run APP/Home.py
# python -m streamlit run APP/Home.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None

# TensorFlow es opcional: si no existe, la app usa motor de reglas.
try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None

# ------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATA"
PROCESSED_DIR = DATA_DIR / "PROCESSED"
RAW_DIR = DATA_DIR / "RAW"
MODELS_DIR = ROOT / "MODELS"
DEFAULT_API_URL = os.getenv("ACADEMIC_API_URL", "http://127.0.0.1:8000/predict")
API_TIMEOUT_SECONDS = int(os.getenv("ACADEMIC_API_TIMEOUT", "15"))

st.set_page_config(
    page_title="Alerta Académica IA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Estilos visuales
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .main {background-color: #f6f8fb;}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {
        background: linear-gradient(135deg, #102a43 0%, #243b53 55%, #486581 100%);
        color: white; padding: 28px 32px; border-radius: 22px;
        box-shadow: 0 10px 28px rgba(16, 42, 67, .18);
    }
    .hero h1 {font-size: 2.15rem; margin-bottom: .25rem;}
    .hero p {font-size: 1.02rem; opacity: .95; margin-bottom: 0;}
    .card {
        background: white; padding: 20px; border-radius: 18px;
        border: 1px solid #e5eaf0; box-shadow: 0 6px 18px rgba(16, 42, 67, .07);
    }
    .risk-high {border-left: 8px solid #d64545;}
    .risk-mid {border-left: 8px solid #f0b429;}
    .risk-low {border-left: 8px solid #2f855a;}
    .small-muted {color: #627d98; font-size: .92rem;}
    div[data-testid="stMetricValue"] {font-size: 1.65rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Utilidades de datos
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def create_demo_data(n: int = 120) -> pd.DataFrame:
    """Dataset demo por si el repositorio aún no tiene CSV procesado."""
    rng = np.random.default_rng(42)
    asistencia = rng.normal(78, 14, n).clip(35, 100).round(1)
    promedio = rng.normal(72, 13, n).clip(25, 100).round(1)
    tareas = rng.normal(76, 18, n).clip(20, 100).round(1)
    participacion = rng.normal(68, 20, n).clip(10, 100).round(1)
    reprobadas = rng.poisson(1.1, n).clip(0, 6)
    ausencias = rng.poisson(5, n).clip(0, 25)
    apoyo = rng.integers(1, 6, n)
    horas_estudio = rng.normal(8, 4, n).clip(0, 25).round(1)

    score = (
        0.32 * promedio
        + 0.24 * asistencia
        + 0.16 * tareas
        + 0.10 * participacion
        + 0.08 * horas_estudio * 4
        + 0.06 * apoyo * 20
        - 5.2 * reprobadas
        - 0.85 * ausencias
    ).clip(0, 100)

    riesgo = pd.cut(
        score,
        bins=[-1, 55, 72, 100],
        labels=["Sin riesgo", "Riesgo bajo", "Riesgo medio", "Riesgo alto"],
    ).astype(str)

    return pd.DataFrame(
        {
            "id_estudiante": [f"EST-{i:03d}" for i in range(1, n + 1)],
            "carrera": rng.choice(["Informática", "Administración", "Salud", "Ingeniería", "Educación"], n),
            "nivel": rng.choice(["I", "II", "III", "IV"], n),
            "asistencia": asistencia,
            "promedio_actual": promedio,
            "entrega_tareas": tareas,
            "participacion": participacion,
            "materias_reprobadas": reprobadas,
            "ausencias": ausencias,
            "apoyo_familiar": apoyo,
            "horas_estudio_semana": horas_estudio,
            "puntaje_riesgo": (100 - score).round(1),
            "riesgo_desercion": riesgo,
            "rendimiento_predicho": score.round(1),
        }
    )


@st.cache_data(show_spinner=False)
def load_repository_data() -> pd.DataFrame:
    """Busca CSV/XLSX en DATA/PROCESSED o DATA/RAW. Si no encuentra, usa demo."""
    candidates: List[Path] = []
    for folder in [PROCESSED_DIR, RAW_DIR, DATA_DIR]:
        if folder.exists():
            candidates.extend(folder.glob("*.csv"))
            candidates.extend(folder.glob("*.xlsx"))
            candidates.extend(folder.glob("*.xls"))

    if not candidates:
        return create_demo_data()

    file = candidates[0]
    try:
        if file.suffix.lower() == ".csv":
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception:
        return create_demo_data()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres frecuentes para que el dashboard no dependa de un único CSV."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    aliases = {
        "student_id": "id_estudiante",
        "id": "id_estudiante",
        "attendance": "asistencia",
        "absence": "ausencias",
        "absences": "ausencias",
        "grade": "promedio_actual",
        "grades": "promedio_actual",
        "average": "promedio_actual",
        "gpa": "promedio_actual",
        "study_hours": "horas_estudio_semana",
        "failures": "materias_reprobadas",
        "failed_subjects": "materias_reprobadas",
        "participation": "participacion",
        "assignment_completion": "entrega_tareas",
        "tasks": "entrega_tareas",
    }
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})

    defaults = {
        "id_estudiante": [f"EST-{i:03d}" for i in range(1, len(df) + 1)],
        "carrera": "Sin clasificar",
        "nivel": "Sin nivel",
        "asistencia": 75,
        "promedio_actual": 70,
        "entrega_tareas": 75,
        "participacion": 65,
        "materias_reprobadas": 0,
        "ausencias": 0,
        "apoyo_familiar": 3,
        "horas_estudio_semana": 8,
    }
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value

    numeric_cols = [
        "asistencia", "promedio_actual", "entrega_tareas", "participacion",
        "materias_reprobadas", "ausencias", "apoyo_familiar", "horas_estudio_semana"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(defaults[col])

    return df


# ------------------------------------------------------------
# Motor predictivo y reglas de alerta
# ------------------------------------------------------------
def calculate_academic_score(row: pd.Series) -> float:
    """Modelo de respaldo interpretable. Rango 0-100."""
    score = (
        0.33 * row["promedio_actual"]
        + 0.23 * row["asistencia"]
        + 0.16 * row["entrega_tareas"]
        + 0.10 * row["participacion"]
        + 0.07 * min(row["horas_estudio_semana"] * 4, 100)
        + 0.06 * row["apoyo_familiar"] * 20
        - 5.0 * row["materias_reprobadas"]
        - 0.75 * row["ausencias"]
    )
    return float(np.clip(score, 0, 100))


def classify_risk(risk_score: float) -> str:
    if risk_score >= 65:
        return "Alto"
    if risk_score >= 40:
        return "Medio"
    return "Bajo"


def risk_badge(risk: str) -> str:
    return {"Alto": "🔴 Alto", "Medio": "🟡 Medio", "Bajo": "🟢 Bajo"}.get(risk, risk)


def generate_recommendations(row: pd.Series) -> List[str]:
    recs: List[str] = []
    if row["asistencia"] < 70:
        recs.append("Activar seguimiento de asistencia: contacto semanal y justificación de ausencias.")
    if row["promedio_actual"] < 65:
        recs.append("Asignar tutoría académica en las materias con menor rendimiento.")
    if row["entrega_tareas"] < 70:
        recs.append("Crear plan de entregas: fechas cortas, revisión de avances y recordatorios.")
    if row["participacion"] < 55:
        recs.append("Derivar a orientación/docente guía para revisar motivación y adaptación al curso.")
    if row["materias_reprobadas"] >= 2:
        recs.append("Revisar carga académica y valorar matrícula reducida o acompañamiento intensivo.")
    if row["horas_estudio_semana"] < 5:
        recs.append("Recomendar horario mínimo de estudio de 6 a 8 horas semanales con técnica Pomodoro.")
    if row["apoyo_familiar"] <= 2:
        recs.append("Coordinar apoyo institucional: becas, orientación o acompañamiento psicoeducativo.")
    if not recs:
        recs.append("Mantener seguimiento preventivo mensual y reforzar hábitos actuales.")
    return recs



# ------------------------------------------------------------
# Cliente API: conexión con modelos ya entrenados
# ------------------------------------------------------------
def build_api_payload(row: pd.Series) -> Dict[str, Any]:
    """Convierte el formulario/dataset al JSON que se enviará al API."""
    return {
        "id_estudiante": str(row.get("id_estudiante", "EST-NUEVO")),
        "carrera": str(row.get("carrera", "General")),
        "nivel": str(row.get("nivel", "I")),
        "asistencia": float(row.get("asistencia", 0)),
        "promedio_actual": float(row.get("promedio_actual", 0)),
        "entrega_tareas": float(row.get("entrega_tareas", 0)),
        "participacion": float(row.get("participacion", 0)),
        "materias_reprobadas": int(row.get("materias_reprobadas", 0)),
        "ausencias": int(row.get("ausencias", 0)),
        "apoyo_familiar": int(row.get("apoyo_familiar", 3)),
        "horas_estudio_semana": float(row.get("horas_estudio_semana", 0)),
    }


def call_prediction_api(payload: Dict[str, Any], api_url: str, timeout: int = API_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Envía un estudiante al API del modelo y devuelve el JSON de respuesta."""
    if requests is None:
        raise RuntimeError("La librería 'requests' no está instalada. Ejecute: python -m pip install requests")

    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_api_response(api_response: Dict[str, Any], row: pd.Series) -> pd.Series:
    """
    Acepta varias formas de respuesta del API para no depender de un único formato.

    Formato recomendado del API:
    {
      "rendimiento_predicho": 74.5,
      "puntaje_riesgo": 25.5,
      "riesgo_desercion": "Bajo",
      "recomendaciones": ["Mantener seguimiento mensual"]
    }
    """
    parsed = row.copy()

    rendimiento = api_response.get("rendimiento_predicho")
    if rendimiento is None:
        rendimiento = api_response.get("rendimiento")
    if rendimiento is None:
        rendimiento = api_response.get("prediction")
    if rendimiento is None:
        rendimiento = api_response.get("prediccion")

    prob_riesgo = api_response.get("probabilidad_riesgo")
    if prob_riesgo is None:
        prob_riesgo = api_response.get("risk_probability")
    if prob_riesgo is None:
        prob_riesgo = api_response.get("probability")

    puntaje_riesgo = api_response.get("puntaje_riesgo")
    if puntaje_riesgo is None:
        puntaje_riesgo = api_response.get("risk_score")

    riesgo = api_response.get("riesgo_desercion")
    if riesgo is None:
        riesgo = api_response.get("riesgo")
    if riesgo is None:
        riesgo = api_response.get("risk_level")

    # Interpretación flexible:
    # - Si el API devuelve rendimiento 0-1, se escala a 0-100.
    # - Si devuelve probabilidad de riesgo 0-1, se escala a 0-100.
    if rendimiento is not None:
        rendimiento = float(rendimiento)
        if 0 <= rendimiento <= 1:
            rendimiento *= 100
        parsed["rendimiento_predicho"] = round(float(np.clip(rendimiento, 0, 100)), 1)

    if puntaje_riesgo is not None:
        puntaje_riesgo = float(puntaje_riesgo)
        if 0 <= puntaje_riesgo <= 1:
            puntaje_riesgo *= 100
        parsed["puntaje_riesgo"] = round(float(np.clip(puntaje_riesgo, 0, 100)), 1)
    elif prob_riesgo is not None:
        prob_riesgo = float(prob_riesgo)
        if 0 <= prob_riesgo <= 1:
            prob_riesgo *= 100
        parsed["puntaje_riesgo"] = round(float(np.clip(prob_riesgo, 0, 100)), 1)
    elif rendimiento is not None:
        parsed["puntaje_riesgo"] = round(100 - parsed["rendimiento_predicho"], 1)

    if riesgo:
        riesgo = str(riesgo).strip().capitalize()
        if riesgo in ["High", "Alto", "1"]:
            riesgo = "Alto"
        elif riesgo in ["Medium", "Medio", "Moderado", "2"]:
            riesgo = "Medio"
        elif riesgo in ["Low", "Bajo", "0"]:
            riesgo = "Bajo"
        parsed["riesgo_desercion"] = riesgo
    elif "puntaje_riesgo" in parsed:
        parsed["riesgo_desercion"] = classify_risk(float(parsed["puntaje_riesgo"]))

    if "rendimiento_predicho" not in parsed and "puntaje_riesgo" in parsed:
        parsed["rendimiento_predicho"] = round(100 - float(parsed["puntaje_riesgo"]), 1)

    alerta = api_response.get("alerta_temprana") or api_response.get("alerta")
    if alerta:
        parsed["alerta_temprana"] = str(alerta)
    else:
        parsed["alerta_temprana"] = {
            "Alto": "Intervención urgente",
            "Medio": "Seguimiento preventivo",
            "Bajo": "Sin alerta crítica",
        }.get(parsed.get("riesgo_desercion", "Medio"), "Seguimiento preventivo")

    recomendaciones = api_response.get("recomendaciones") or api_response.get("recommendations")
    if isinstance(recomendaciones, list):
        parsed["recomendaciones"] = " | ".join(str(r) for r in recomendaciones)
    elif isinstance(recomendaciones, str):
        parsed["recomendaciones"] = recomendaciones
    else:
        parsed["recomendaciones"] = " | ".join(generate_recommendations(parsed))

    parsed["fuente_prediccion"] = "API"
    return parsed


def predict_with_api_or_rules(row: pd.Series, api_url: str, use_api: bool = True) -> Tuple[pd.Series, str | None]:
    """Predice usando API. Si falla, usa el motor de reglas y devuelve el error."""
    base = row.copy()
    if use_api:
        try:
            payload = build_api_payload(base)
            api_response = call_prediction_api(payload, api_url)
            return parse_api_response(api_response, base), None
        except Exception as exc:
            error_msg = str(exc)
    else:
        error_msg = None

    base["rendimiento_predicho"] = round(calculate_academic_score(base), 1)
    base["puntaje_riesgo"] = round(100 - base["rendimiento_predicho"], 1)
    base["riesgo_desercion"] = classify_risk(base["puntaje_riesgo"])
    base["alerta_temprana"] = {
        "Alto": "Intervención urgente",
        "Medio": "Seguimiento preventivo",
        "Bajo": "Sin alerta crítica",
    }[base["riesgo_desercion"]]
    base["recomendaciones"] = " | ".join(generate_recommendations(base))
    base["fuente_prediccion"] = "Reglas locales"
    return base, error_msg


def enrich_predictions_with_api(df: pd.DataFrame, api_url: str, use_api: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Predice un lote completo. Puede consumir API o usar reglas locales."""
    df = normalize_columns(df)
    rows = []
    errors: List[str] = []
    for _, row in df.iterrows():
        predicted, error = predict_with_api_or_rules(row, api_url=api_url, use_api=use_api)
        rows.append(predicted)
        if error:
            errors.append(error)
    return pd.DataFrame(rows), errors


def enrich_predictions(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df["rendimiento_predicho"] = df.apply(calculate_academic_score, axis=1).round(1)
    df["puntaje_riesgo"] = (100 - df["rendimiento_predicho"]).round(1)
    df["riesgo_desercion"] = df["puntaje_riesgo"].apply(classify_risk)
    df["alerta_temprana"] = df["riesgo_desercion"].map(
        {"Alto": "Intervención urgente", "Medio": "Seguimiento preventivo", "Bajo": "Sin alerta crítica"}
    )
    df["recomendaciones"] = df.apply(lambda r: " | ".join(generate_recommendations(r)), axis=1)
    df["fuente_prediccion"] = "Reglas locales"
    return df


@st.cache_resource(show_spinner=False)
def load_available_model():
    """Carga un modelo Keras si existe. Se deja listo para futura integración."""
    if load_model is None or not MODELS_DIR.exists():
        return None
    for model_name in ["modelo_riesgo_engineering.keras", "modelo_riesgo.keras", "model1_featureengineering.h5", "model1.h5"]:
        path = MODELS_DIR / model_name
        if path.exists():
            try:
                return load_model(path)
            except Exception:
                return None
    return None


# ------------------------------------------------------------
# Componentes UI
# ------------------------------------------------------------
def metric_card(label: str, value: str, delta: str | None = None):
    st.metric(label=label, value=value, delta=delta)


def show_recommendation_panel(row: pd.Series):
    risk_class = {"Alto": "risk-high", "Medio": "risk-mid", "Bajo": "risk-low"}.get(row["riesgo_desercion"], "")
    st.markdown(
        f"""
        <div class="card {risk_class}">
            <h3>Perfil del estudiante: {row['id_estudiante']}</h3>
            <p><b>Riesgo de deserción:</b> {risk_badge(row['riesgo_desercion'])}</p>
            <p><b>Rendimiento académico predicho:</b> {row['rendimiento_predicho']} / 100</p>
            <p><b>Alerta temprana:</b> {row['alerta_temprana']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Recomendaciones automáticas de intervención")
    for rec in generate_recommendations(row):
        st.write(f"- {rec}")


# ------------------------------------------------------------
# App principal
# ------------------------------------------------------------
st.sidebar.title("🎓 Alerta Académica IA")
st.sidebar.caption("Predicción de desempeño y riesgo de deserción")

st.sidebar.subheader("Conexión con API del modelo")
api_url = st.sidebar.text_input("URL del endpoint predict", value=DEFAULT_API_URL)
use_api_batch = st.sidebar.toggle(
    "Usar API para el dashboard completo",
    value=False,
    help="Actívelo cuando su API esté corriendo. Si falla, la app usa reglas locales como respaldo."
)

if st.sidebar.button("Probar conexión API"):
    sample_payload = build_api_payload(normalize_columns(create_demo_data(1)).iloc[0])
    try:
        sample_response = call_prediction_api(sample_payload, api_url)
        st.sidebar.success("API conectada correctamente.")
        st.sidebar.json(sample_response)
    except Exception as exc:
        st.sidebar.error(f"No se pudo conectar al API: {exc}")

page = st.sidebar.radio(
    "Navegación",
    ["Dashboard", "Predicción individual", "Alertas tempranas", "Análisis de datos", "Acerca del sistema"],
)

df_base = load_repository_data()
if use_api_batch:
    with st.spinner("Consumiendo API para generar predicciones del dashboard..."):
        df, api_errors = enrich_predictions_with_api(df_base, api_url=api_url, use_api=True)
    if api_errors:
        st.warning("Algunas predicciones no pudieron obtenerse desde el API. Se usó respaldo local en esos casos.")
else:
    df = enrich_predictions(df_base)
    api_errors = []

model = load_available_model()

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")
careers = sorted(df["carrera"].astype(str).unique().tolist())
selected_careers = st.sidebar.multiselect("Carrera", careers, default=careers)
selected_risk = st.sidebar.multiselect("Nivel de riesgo", ["Alto", "Medio", "Bajo"], default=["Alto", "Medio", "Bajo"])

filtered = df[df["carrera"].astype(str).isin(selected_careers) & df["riesgo_desercion"].isin(selected_risk)]

st.markdown(
    """
    <div class="hero">
        <h1>Sistema de Predicción de Rendimiento Académico</h1>
        <p>Dashboard inteligente para detectar riesgo de deserción, generar alertas tempranas y recomendar intervenciones educativas oportunas.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if page == "Dashboard":
    total = len(filtered)
    high = int((filtered["riesgo_desercion"] == "Alto").sum())
    mid = int((filtered["riesgo_desercion"] == "Medio").sum())
    low = int((filtered["riesgo_desercion"] == "Bajo").sum())
    avg_perf = filtered["rendimiento_predicho"].mean() if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Estudiantes analizados", f"{total}")
    with c2:
        metric_card("Riesgo alto", f"{high}", f"{(high / total * 100):.1f}%" if total else "0%")
    with c3:
        metric_card("Riesgo medio", f"{mid}", f"{(mid / total * 100):.1f}%" if total else "0%")
    with c4:
        metric_card("Rendimiento promedio", f"{avg_perf:.1f}/100")

    st.markdown("---")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Distribución de riesgo")
        if px:
            risk_counts = filtered["riesgo_desercion"].value_counts().reindex(["Alto", "Medio", "Bajo"]).fillna(0).reset_index()
            risk_counts.columns = ["riesgo", "cantidad"]
            fig = px.bar(risk_counts, x="riesgo", y="cantidad", text="cantidad", title="Estudiantes por nivel de riesgo")
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(filtered["riesgo_desercion"].value_counts())

    with right:
        st.subheader("Rendimiento vs asistencia")
        if px:
            fig = px.scatter(
                filtered,
                x="asistencia",
                y="rendimiento_predicho",
                color="riesgo_desercion",
                hover_data=["id_estudiante", "carrera", "promedio_actual"],
                title="Relación entre asistencia y desempeño predicho",
            )
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.scatter_chart(filtered, x="asistencia", y="rendimiento_predicho")

    st.subheader("Top estudiantes que requieren atención")
    cols = ["id_estudiante", "carrera", "nivel", "puntaje_riesgo", "riesgo_desercion", "alerta_temprana", "rendimiento_predicho", "fuente_prediccion"]
    st.dataframe(
        filtered.sort_values("puntaje_riesgo", ascending=False)[cols].head(15),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Predicción individual":
    st.subheader("Evaluación individual del estudiante")
    st.caption("Ingrese los indicadores actuales para estimar desempeño, riesgo y plan de intervención.")

    use_api_single = st.checkbox("Consumir API para esta predicción", value=True)

    with st.form("student_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            student_id = st.text_input("ID del estudiante", "EST-NUEVO")
            carrera = st.selectbox("Carrera", careers if careers else ["General"])
            nivel = st.selectbox("Nivel", ["I", "II", "III", "IV", "V"])
        with col2:
            asistencia = st.slider("Asistencia (%)", 0, 100, 78)
            promedio = st.slider("Promedio actual", 0, 100, 72)
            tareas = st.slider("Entrega de tareas (%)", 0, 100, 75)
        with col3:
            participacion = st.slider("Participación (%)", 0, 100, 65)
            reprobadas = st.number_input("Materias reprobadas", min_value=0, max_value=10, value=1)
            ausencias = st.number_input("Ausencias", min_value=0, max_value=40, value=4)
            apoyo = st.slider("Apoyo familiar/institucional (1-5)", 1, 5, 3)
            horas = st.slider("Horas de estudio por semana", 0, 30, 8)

        submitted = st.form_submit_button("Generar predicción")

    if submitted:
        row = pd.Series(
            {
                "id_estudiante": student_id,
                "carrera": carrera,
                "nivel": nivel,
                "asistencia": asistencia,
                "promedio_actual": promedio,
                "entrega_tareas": tareas,
                "participacion": participacion,
                "materias_reprobadas": reprobadas,
                "ausencias": ausencias,
                "apoyo_familiar": apoyo,
                "horas_estudio_semana": horas,
            }
        )
        row, api_error = predict_with_api_or_rules(row, api_url=api_url, use_api=use_api_single)
        if api_error and use_api_single:
            st.warning(f"No se pudo consumir el API. Se usó el respaldo local. Detalle: {api_error}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rendimiento predicho", f"{row['rendimiento_predicho']}/100")
        c2.metric("Puntaje de riesgo", f"{row['puntaje_riesgo']}/100")
        c3.metric("Riesgo", risk_badge(row["riesgo_desercion"]))
        st.caption(f"Fuente de predicción: {row.get('fuente_prediccion', 'No especificada')}")
        show_recommendation_panel(row)

elif page == "Alertas tempranas":
    st.subheader("Sistema de alertas tempranas")
    st.caption("Priorización automática para intervención educativa.")

    alert_df = filtered.sort_values("puntaje_riesgo", ascending=False).copy()
    alert_df["prioridad"] = np.select(
        [alert_df["riesgo_desercion"].eq("Alto"), alert_df["riesgo_desercion"].eq("Medio")],
        ["1 - Urgente", "2 - Preventiva"],
        default="3 - Monitoreo",
    )

    st.dataframe(
        alert_df[[
            "prioridad", "id_estudiante", "carrera", "nivel", "puntaje_riesgo", "riesgo_desercion",
            "asistencia", "promedio_actual", "materias_reprobadas", "alerta_temprana", "recomendaciones", "fuente_prediccion"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    csv = alert_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar reporte de alertas CSV",
        data=csv,
        file_name="reporte_alertas_tempranas.csv",
        mime="text/csv",
    )

elif page == "Análisis de datos":
    st.subheader("Análisis exploratorio")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Estadísticas principales")
        st.dataframe(filtered[["asistencia", "promedio_actual", "entrega_tareas", "participacion", "rendimiento_predicho", "puntaje_riesgo"]].describe().round(2), use_container_width=True)
    with c2:
        st.markdown("#### Riesgo por carrera")
        table = pd.crosstab(filtered["carrera"], filtered["riesgo_desercion"])
        st.dataframe(table, use_container_width=True)

    st.markdown("#### Dataset utilizado")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

elif page == "Acerca del sistema":
    st.subheader("Acerca del sistema")
    st.markdown(
        """
        Este prototipo cumple con cuatro funciones principales:

        1. **Predice el rendimiento académico** mediante indicadores de asistencia, promedio, tareas, participación, materias reprobadas, ausencias, apoyo y horas de estudio.
        2. **Detecta riesgo de deserción** clasificando a cada estudiante en bajo, medio o alto.
        3. **Genera alertas tempranas** para priorizar casos que necesitan intervención.
        4. **Recomienda acciones automáticas** según el perfil del estudiante.

        El sistema está preparado para consumir una API de predicción. Streamlit envía los indicadores del estudiante al endpoint configurado en el menú lateral y recibe el rendimiento predicho, riesgo de deserción, alerta temprana y recomendaciones. Si la API no está disponible, utiliza un motor de reglas interpretable como respaldo para que la aplicación siga funcionando durante la presentación.
        """
    )
    if model is not None:
        st.success("Modelo Keras detectado y cargado correctamente.")
    else:
        st.warning("No se cargó un modelo Keras compatible. La app está usando el motor de reglas demostrativo.")
