import streamlit as st
import pandas as pd
import joblib

# 1. Cargar el pipeline completo
@st.cache_resource
def load_pipeline():
    modelo = joblib.load('modelo_gradient_boosting.pkl')
    escalador = joblib.load('escalador.pkl')
    cols_num = joblib.load('columnas_numericas.pkl')
    cols_train = joblib.load('columnas_entrenamiento.pkl')
    return modelo, escalador, cols_num, cols_train

modelo, escalador, columnas_numericas, columnas_entrenamiento = load_pipeline()

st.title("🎓 Predicción de Puntaje Global - Saber Pro UNAL")
st.write("Esta aplicación utiliza un modelo de Machine Learning (Gradient Boosting) para estimar el puntaje global de un estudiante.")

# 2. Interfaz de recolección de datos
st.sidebar.header("Perfil del Estudiante")

# Variables numéricas estándar
edad = st.sidebar.slider("EDAD", 16, 65, 23)

# Variables categóricas estándar del contexto universitario
estrato = st.sidebar.selectbox("ESTRATO", ["Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"])
sexo = st.sidebar.selectbox("SEXO", ["Hombre", "Mujer"])
sede = st.sidebar.selectbox("SEDE_ADMISION", ["BOGOTÁ", "MEDELLÍN", "MANIZALES", "PALMIRA", "AMAZONIA", "CARIBE", "ORINOQUIA", "TUMACO"])
colegio = st.sidebar.selectbox("TIPO_COLEGIO", ["OFICIAL", "NO OFICIAL"])
facultad = st.sidebar.selectbox("FACULTAD", ["INGENIERIA", "MINAS", "CIENCIAS HUMANAS", "CIENCIAS", "ARTES", "DERECHO", "MEDICINA"])

if st.button("Realizar Predicción"):
    # 3. Construir el dato crudo (Agrega aquí más columnas si las tenías en el dataset original antes del get_dummies)
    nuevo_dato = pd.DataFrame({
        'EDAD': [edad],
        'ESTRATO': [estrato],
        'SEXO': [sexo],
        'SEDE_ADMISION': [sede],
        'TIPO_COLEGIO': [colegio],
        'FACULTAD': [facultad]
    })
    
    # 4. Escalar SOLO las variables numéricas que el modelo normalizó originalmente
    for col in columnas_numericas:
        if col not in nuevo_dato.columns:
            nuevo_dato[col] = 0.0 # Rellena de seguridad si falta una métrica en la interfaz
            
    nuevo_dato[columnas_numericas] = escalador.transform(nuevo_dato[columnas_numericas])
    
    # 5. Generar Dummies y alinear estrictamente con las columnas de entrenamiento
    nuevo_dato_dummies = pd.get_dummies(nuevo_dato)
    nuevo_dato_alineado = nuevo_dato_dummies.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # 6. Predicción
    try:
        prediccion = modelo.predict(nuevo_dato_alineado)
        st.success(f"El puntaje global estimado es: **{prediccion[0]:.2f}**")
        st.metric(label="Rango Esperado (RMSE ±18.8)", value=f"{prediccion[0]:.0f}", delta="Puntos Saber Pro")
    except Exception as e:
        st.error(f"Error técnico de alineación: {e}")
