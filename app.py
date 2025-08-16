import streamlit as st
import pandas as pd
import pickle
from pycaret.classification import load_model, predict_model
import numpy as np

# Cargar el modelo usando caching para evitar recargar en cada interacción
@st.cache_resource
def get_model():
    return load_model('titanic_model')

# Definir la función predict usando pycaret
def predict(model, df):
    # Obtener predicciones usando pycaret
    predictions = predict_model(model, data=df)
    # Adaptar el nombre de la columna según la versión de pycaret
    if 'score' in predictions.columns:
        return predictions['score'][0]
    elif 'prediction_score' in predictions.columns:
        return predictions['prediction_score'][0]
    else:
        raise ValueError("No se encontró la columna de predicción esperada en los resultados.")

# Cargar el modelo
model = get_model()

# Definir categorías de salario y functional area
Passenger_categories = ['First', 'Second', 'Third']
sex_categories = ["Male","Female"]

with st.form("form"):
    st.header("Ingrese los datos del Pasajero")
    
    
    # Slider para 'Edad de pasajero'
    Age = st.slider('Edad de pasajero', min_value=0.0, max_value=80.0, value=0.0, step=0.1)
    
    # Slider para 'Cantidad de parientes'
    Hermanos_o_Esposas = st.slider('Hermanos o espos@', min_value=0, max_value=8, value=0, step=1)
    
    # Slider para 'Niños'
    Niños = st.slider('Niños', min_value=0, max_value=9, value=0, step=1)
    
    # Slider para 'Tarifa'
    Tarifa = st.slider('Tarifa', min_value=0.0, max_value=512.3, value=0.0, step=0.1)

    # Selectbox para 'Número de pasajero'
    Passenger = st.selectbox('Número de pasajero', Passenger_categories)
    
    # Selectbox para 'Sexo'
    Sex = st.selectbox("Sexo", sex_categories)
    
    # Botón para predecir
    predict_button = st.form_submit_button('Predecir')
    
   

# Crear un DataFrame con las columnas esperadas para la predicción
input_dict = {

    'Age': Age,
    'Hermanos o Esposas': Hermanos_o_Esposas,
    'Niños': Niños,
    'Tarifa': Tarifa,
    "Passenger": Passenger,
    "Sex": Sex,
    'Passenger_First': 1 if Passenger == 'First' else 0,
    'Passenger_Second': 1 if Passenger == 'Second' else 0,
    'Passenger_Third': 1 if Passenger == 'Third' else 0,
    'Sex_Male': 1 if Sex == 'Male' else 0,
    'Sex_Female': 1 if Sex == 'Female' else 0,
}

input_df = pd.DataFrame([input_dict])

# Mostrar el DataFrame con los datos de entrada
st.write("Datos de Entrada del Pasajero:")
st.dataframe(input_df, use_container_width=True)

try:
    # Hacer la predicción y obtener la probabilidad
    prediction_score = predict(model, input_df)

    # Mostrar la probabilidad y el resultado de la predicción
    st.write(f"Probabilidad de Sobrevivir: {prediction_score:.2f}")
    if prediction_score > 0.5:  
        st.success('El pasajero tiene más del 50 % de sobrevivir.')
    else:
        st.success('El pasajero tiene menos del 50 % de sobrevivir.')

except Exception as e:
    st.error(f'Ocurrió un error: {e}')
