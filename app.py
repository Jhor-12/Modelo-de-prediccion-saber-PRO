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

# El año se ajusta como categoría cerrada según los datos de entrenamiento
year = st.sidebar.selectbox("AÑO (YEAR)", [2018, 2019, 2020, 2021, 2022])

# Variables numéricas
edad = st.sidebar.slider("EDAD", 15, 80, 23)
pbm = st.sidebar.number_input("PBM (Puntaje Básico de Matrícula)", min_value=0.0, max_value=100.0, value=50.0)

# Variables categóricas
nacionalidad = st.sidebar.selectbox("NACIONALIDAD", ["COLOMBIA", "EXTRANJERO"]) 
sexo = st.sidebar.selectbox("SEXO", ["Hombre", "Mujer"])
estrato = st.sidebar.selectbox("ESTRATO", ["Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"])
tipo_col = st.sidebar.selectbox("TIPO_COL (Tipo de Colegio)", ["OFICIAL", "NO OFICIAL"])

sedes_unal = ["BOGOTÁ", "MEDELLÍN", "MANIZALES", "PALMIRA", "AMAZONIA", "CARIBE", "ORINOQUIA", "TUMACO", "LA PAZ"]
sede_admision = st.sidebar.selectbox("SEDE_ADMISION", sedes_unal)
sede_finalizacion = st.sidebar.selectbox("SEDE_FINALIZACION", sedes_unal)

mod_adm = st.sidebar.selectbox("MOD_ADM (Modalidad Admisión)", ["REGULAR", "PEAMA", "PAES"])
tipo_adm = st.sidebar.selectbox("TIPO_ADM", ["REGULAR", "ESPECIAL"])
paes = st.sidebar.selectbox("PAES", ["SI", "NO"])
peama = st.sidebar.selectbox("PEAMA", ["SI", "NO"])

facultad = st.sidebar.selectbox("FACULTAD", ["INGENIERIA", "MINAS", "CIENCIAS HUMANAS", "CIENCIAS", "ARTES", "DERECHO", "MEDICINA"])
areac_snies = st.sidebar.selectbox("AREAC_SNIES", ["INGENIERIA, ARQUITECTURA", "CIENCIAS SOCIALES", "MATEMATICAS", "ARTES"]) 
area_cine = st.sidebar.selectbox("AREA_CINE", ["INGENIERIA", "ARTES", "CIENCIAS"])

if st.button("Realizar Predicción"):
    # 3. Construir el dato crudo con las 16 variables requeridas
    nuevo_dato = pd.DataFrame({
        'YEAR': [year],
        'NACIONALIDAD': [nacionalidad],
        'EDAD': [edad],
        'SEXO': [sexo],
        'ESTRATO': [estrato],
        'TIPO_COL': [tipo_col],
        'PBM': [pbm],
        'SEDE_ADMISION': [sede_admision],
        'SEDE_FINALIZACION': [sede_finalizacion],
        'MOD_ADM': [mod_adm],
        'TIPO_ADM': [tipo_adm],
        'PAES': [paes],
        'PEAMA': [peama],
        'FACULTAD': [facultad],
        'AREAC_SNIES': [areac_snies],
        'AREA_CINE': [area_cine]
    })
    
    # 4. Generar Dummies (convertir texto a columnas) PRIMERO
    nuevo_dato_dummies = pd.get_dummies(nuevo_dato, columns=['YEAR', 'NACIONALIDAD', 'SEXO', 'ESTRATO', 'TIPO_COL', 'SEDE_ADMISION', 'SEDE_FINALIZACION', 'MOD_ADM', 'TIPO_ADM', 'PAES', 'PEAMA', 'FACULTAD', 'AREAC_SNIES', 'AREA_CINE'] if 'YEAR' in nuevo_dato.columns else None)
    
    # 5. Alinear estrictamente con las columnas de entrenamiento
    nuevo_dato_alineado = nuevo_dato_dummies.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # 6. Escalar los datos DESPUÉS de alinear
    # Así nos aseguramos de que cualquier columna (original o dummy) se escale correctamente
    nuevo_dato_alineado[columnas_numericas] = escalador.transform(nuevo_dato_alineado[columnas_numericas])
    
    # 7. Predicción
    try:
        prediccion = modelo.predict(nuevo_dato_alineado)
        st.success(f"El puntaje global estimado es: **{prediccion[0]:.2f}**")
        st.metric(label="Rango Esperado (RMSE ±18.8)", value=f"{prediccion[0]:.0f}", delta="Puntos Saber Pro")
    except Exception as e:
        st.error(f"Error técnico de predicción: {e}")
