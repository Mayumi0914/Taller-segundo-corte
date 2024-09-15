import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model

# Cargar el modelo preentrenado
with open('best_model.pkl', 'rb') as model_file:
    modelo = pickle.load(model_file)

# Cargar los datos de prueba
prueba = pd.read_csv("prueba_APP.csv")
# Imprimir las columnas para verificar

# Convertir las columnas categóricas en objetos si existen
ct = ["dominio", "Tec"]
for k in ct:
    if k in prueba.columns:
        prueba[k] = prueba[k].astype("O")

# Título de la API
st.title("API de Predicción precio")

# Entradas del usuario para los selectbox en el orden de la imagen
dominio = st.selectbox("dominio", ['yahoo', 'Otro', 'gmail', 'hotmail'])
Tec = st.selectbox("Tec", ['PC', 'Smartphone', 'Iphone', 'Portatil'])
Avg_Session_Length = st.text_input("Avg. Session Length", value="32.06")
Time_on_App = st.text_input("Time on App", value="10.7")
Time_on_Website = st.text_input("Time on Website", value="37.71")
Length_of_Membership = st.text_input("Length of Membership", value="3.004")


# Convertir los valores de texto a números si es posible
if st.button("Calcular"):
    try:
        Avg_Session_Length = float(Avg_Session_Length)
        Time_on_App = float(Time_on_App)
        Time_on_Website = float(Time_on_Website)
        Length_of_Membership = float(Length_of_Membership)
        

        # Crear el dataframe a partir de los inputs del usuario
        user = pd.DataFrame({
            'dominio': [dominio],
            'Tec': [Tec],
            'Avg. Session Length': [Avg_Session_Length],
            'Time on App': [Time_on_App],
            'Time on Website': [Time_on_Website],
            'Length of Membership': [Length_of_Membership],
            
        })

        # Hacer predicciones utilizando el modelo cargado
        predictions = predict_model(modelo, data=user)

        # Mostrar la predicción al usuario
        st.write(f'La predicción es: {predictions["prediction_label"][0]}')

    except ValueError:
        st.error("Por favor, ingrese valores numéricos válidos en los campos correspondientes.")

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.experimental_rerun()