"""
Script para descargar Student Performance Dataset
Dataset: https://archive.ics.uci.edu/dataset/320/student+performance

Ejecutar desde raíz: python data/raw/download_data.py
"""

import urllib.request
import zipfile
import os
import pandas as pd

# URLs del dataset
DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def download_file(url, destination):
    """Descarga archivo desde URL"""
    print(f"Descargando {url}...")
    urllib.request.urlretrieve(url, destination)
    print(f"✓ Descargado: {destination}")

def extract_zip(zip_path, extract_to):
    """Extrae archivo ZIP"""
    print(f"Extrayendo {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extraído a: {extract_to}")

def process_student_data():
    """Procesa los datasets de estudiantes"""
    # Hay dos archivos: student-mat.csv (matemáticas) y student-por.csv (portugués)
    
    # Matemáticas
    mat_path = os.path.join(RAW_DIR, "student-mat.csv")
    por_path = os.path.join(RAW_DIR, "student-por.csv")
    
    print("\n" + "="*60)
    print("PROCESANDO DATASETS")
    print("="*60)
    
    # Cargar matemáticas
    print("\n--- Dataset de Matemáticas ---")
    df_mat = pd.read_csv(mat_path, sep=';')
    df_mat['subject'] = 'math'
    print(f"Registros: {df_mat.shape[0]}, Columnas: {df_mat.shape[1]}")
    
    # Cargar portugués
    print("\n--- Dataset de Portugués ---")
    df_por = pd.read_csv(por_path, sep=';')
    df_por['subject'] = 'portuguese'
    print(f"Registros: {df_por.shape[0]}, Columnas: {df_por.shape[1]}")
    
    # Combinar ambos datasets
    print("\n--- Combinando datasets ---")
    df_combined = pd.concat([df_mat, df_por], ignore_index=True)
    print(f"Total registros: {df_combined.shape[0]}")
    
    print("\nColumnas del dataset:")
    print(df_combined.columns.tolist())
    
    print("\nPrimeras filas:")
    print(df_combined.head())
    
    print("\nInformación del dataset:")
    print(df_combined.info())
    
    print("\nEstadísticas de calificación final (G3):")
    print(df_combined['G3'].describe())
    
    print("\nDistribución por materia:")
    print(df_combined['subject'].value_counts())
    
    # Guardar datasets procesados
    # Individual por materia
    mat_processed = os.path.join(PROCESSED_DIR, "student_math.csv")
    por_processed = os.path.join(PROCESSED_DIR, "student_portuguese.csv")
    combined_processed = os.path.join(PROCESSED_DIR, "student_performance.csv")
    
    df_mat.to_csv(mat_processed, index=False)
    df_por.to_csv(por_processed, index=False)
    df_combined.to_csv(combined_processed, index=False)
    
    print(f"\n✓ Datasets guardados:")
    print(f"  - Matemáticas: {mat_processed}")
    print(f"  - Portugués: {por_processed}")
    print(f"  - Combinado: {combined_processed}")
    
    # Análisis de riesgo académico
    print("\n--- Análisis de Riesgo Académico ---")
    # Considerar G3 < 10 como riesgo de reprobación
    risk_threshold = 10
    df_combined['at_risk'] = df_combined['G3'] < risk_threshold
    
    print(f"Estudiantes en riesgo (G3 < {risk_threshold}): {df_combined['at_risk'].sum()} ({df_combined['at_risk'].mean():.1%})")
    
    return df_combined

def main():
    """Función principal"""
    # Crear directorios
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("="*60)
    print("DESCARGA DE DATASET: Student Performance")
    print("="*60)
    
    # Descargar ZIP
    zip_path = os.path.join(RAW_DIR, "student_performance.zip")
    
    if not os.path.exists(zip_path):
        download_file(DATA_URL, zip_path)
    else:
        print(f"✓ ZIP ya existe: {zip_path}")
    
    # Extraer
    mat_csv = os.path.join(RAW_DIR, "student-mat.csv")
    if not os.path.exists(mat_csv):
        extract_zip(zip_path, RAW_DIR)
    else:
        print(f"✓ Archivos ya extraídos")
    
    # Procesar
    process_student_data()
    
    print("\n" + "="*60)
    print("¡DESCARGA COMPLETADA!")
    print("="*60)
    print("\nSiguiente paso: notebooks/01_EDA_StudentPerformance.ipynb")
    print("\nNotas:")
    print("- El dataset incluye calificaciones de 3 períodos (G1, G2, G3)")
    print("- G3 es la calificación final (objetivo a predecir)")
    print("- Escala: 0-20 puntos")
    print("- Incluye 30+ variables sobre estudiantes")

if __name__ == "__main__":
    main()
