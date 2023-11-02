import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import numpy as np
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #e5e5f7;
opacity: 0.;
background-image:  radial-gradient(#444cf7 0.5px, transparent 0.5px), radial-gradient(#444cf7 0.5px, #e5e5f7 0.5px);
background-size: 20px 20px;
background-position: 0 0,10px 10px;
</style>
"""



# Agregar una imagen de fondo
st.markdown(page_bg_img, unsafe_allow_html=True)


# Configurar el título de la aplicación
st.title("Explorador de Datos")

# Página 1: Bienvenida y Cargar archivo de datos
st.write("¡Bienvenido a la aplicación de exploración de datos!")
st.write("Esta aplicación te permite cargar un archivo de datos.")

uploaded_file = st.file_uploader("Cargar archivo CSV o Excel (xlsx)", type=["csv", "xlsx"])

# Definir df fuera del bloque if
df = None

if uploaded_file is not None:
    st.write("Archivo cargado con éxito.")
    # Aquí puedes realizar operaciones de exploración de datos con el archivo cargado.
    df = pd.read_csv(uploaded_file)

    # Mostrar información básica sobre el conjunto de datos
    st.subheader("Información del Conjunto de Datos:")
    st.write(f"Número de Filas: {df.shape[0]}")
    st.write(f"Número de Columnas: {df.shape[1]}")
    st.write("Primeras Filas del Conjunto de Datos:")
    st.write(df.head())

# Menú lateral para cambiar entre pestañas
menu_selection = st.sidebar.selectbox("Selecciona una opción", ["Bienvenida y Carga de Archivo", "Análisis Exploratorio", "Combinación de Variables"])

if menu_selection == "Análisis Exploratorio":
    st.write("En esta página se muestra el análisis exploratorio de datos:")

    if df is not None:
        # Listas desplegables con tipos de variables
        st.subheader("Selecciona el tipo de variable:")
        variable_type = st.selectbox("Tipo de Variable:", ["Selecciona un tipo", "Numéricas Continuas", "Numéricas Discretas", "Categóricas", "Tipo Fecha"])

        if variable_type != "Selecciona un tipo":
            if variable_type == "Numéricas Continuas":
                st.subheader("Gráfica de Densidad con Histograma:")
                numeric_continuous_columns = df.select_dtypes(include=['float64']).columns.tolist()
                selected_column = st.selectbox("Selecciona una variable continua:", numeric_continuous_columns)

                if selected_column is not None:
                    # Crear una gráfica de densidad con histograma
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[selected_column], kde=True, ax=ax)
                    st.pyplot(fig)

                    # Calcular y mostrar estadísticas redondeadas
                    st.subheader("Estadísticas:")
                    st.write(f"Media: {round(df[selected_column].mean(), 2)}")
                    st.write(f"Mediana: {round(df[selected_column].median(), 2)}")
                    st.write(f"Desviación Estándar: {round(df[selected_column].std(), 2)}")
                    st.write(f"Varianza: {round(df[selected_column].var(), 2)}")

            elif variable_type == "Numéricas Discretas":
                st.subheader("Gráfica de Histograma:")
                numeric_discrete_columns = df.select_dtypes(include=['int64']).columns.tolist()
                selected_column = st.selectbox("Selecciona una variable discreta:", numeric_discrete_columns)

                if selected_column is not None:
                    # Crear una gráfica de histograma
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[selected_column], ax=ax)
                    st.pyplot(fig)

                    # Calcular y mostrar estadísticas redondeadas
                    st.subheader("Estadísticas:")
                    st.write(f"Media: {round(df[selected_column].mean(), 2)}")
                    st.write(f"Mediana: {round(df[selected_column].median(), 2)}")
                    st.write(f"Desviación Estándar: {round(df[selected_column].std(), 2)}")
                    st.write(f"Varianza: {round(df[selected_column].var(), 2)}")
                    st.write(f"Moda: {df[selected_column].mode().values[0]}")

            elif variable_type == "Categóricas":
                st.subheader("Gráfica de Barras:")
                categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
                selected_column = st.selectbox("Selecciona una variable categórica:", categorical_columns)

                if selected_column is not None:
                    # Crear una gráfica de barras
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=df, x=selected_column, ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                    # Conteo de valores por categoría
                    value_counts = df[selected_column].value_counts()
                    st.subheader("Conteo por Categoría:")
                    st.write(value_counts)

            elif variable_type == "Tipo Fecha":
                st.subheader("Gráfica de Serie de Tiempo:")
                date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
                selected_column = st.selectbox("Selecciona una variable de tipo fecha:", date_columns)

                if selected_column is not None:
                    # Crear una gráfica de serie de tiempo
                    df[selected_column] = pd.to_datetime(df[selected_column])
                    df.set_index(selected_column, inplace=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.plot(df.index, df[selected_column])
                    plt.xlabel("Fecha")
                    plt.ylabel("Valor")
                    st.pyplot(fig)

if menu_selection == "Combinación de Variables":
    st.write("En esta página se muestra la combinación de variables:")

    if df is not None:
        # Selección del tipo de variables
        st.subheader("Selecciona el tipo de variables:")
        variable_type_1 = st.selectbox("Variable 1:", ["Numérica Continua", "Numérica Discreta", "Categórica", "Tipo Fecha"])
        variable_type_2 = st.selectbox("Variable 2:", ["Numérica Continua", "Numérica Discreta", "Categórica", "Tipo Fecha"])

        # Obtener listas de variables de acuerdo al tipo seleccionado
        if variable_type_1 == "Numérica Continua":
            variable_list_1 = df.select_dtypes(include=['float64']).columns.tolist()
        elif variable_type_1 == "Numérica Discreta":
            variable_list_1 = df.select_dtypes(include=['int64']).columns.tolist()
        elif variable_type_1 == "Categórica":
            variable_list_1 = df.select_dtypes(include=['object']).columns.tolist()
        elif variable_type_1 == "Tipo Fecha":
            variable_list_1 = df.select_dtypes(include=['datetime64']).columns.tolist()

        if variable_type_2 == "Numérica Continua":
            variable_list_2 = df.select_dtypes(include=['float64']).columns.tolist()
        elif variable_type_2 == "Numérica Discreta":
            variable_list_2 = df.select_dtypes(include=['int64']).columns.tolist()
        elif variable_type_2 == "Categórica":
            variable_list_2 = df.select_dtypes(include=['object']).columns.tolist()
        elif variable_type_2 == "Tipo Fecha":
            variable_list_2 = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Selección de dos variables
        st.subheader("Selecciona dos variables:")
        selected_variable_1 = st.selectbox("Variable 1:", variable_list_1)
        selected_variable_2 = st.selectbox("Variable 2:", variable_list_2)

        if selected_variable_1 != selected_variable_2:
            # Verificar las combinaciones de variables y realizar visualizaciones y cálculos correspondientes
            if (variable_type_1 == "Numérica Continua" or variable_type_1 == "Numérica Discreta") and (variable_type_2 == "Numérica Continua" or variable_type_2 == "Numérica Discreta"):
                st.subheader("Gráfica de Scatter Plot:")
                # Crear una gráfica de scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df, x=selected_variable_1, y=selected_variable_2, ax=ax)
                st.pyplot(fig)

                # Calcular y mostrar la métrica de correlación
                correlation = df[selected_variable_1].corr(df[selected_variable_2])
                st.subheader("Métrica de Correlación:")
                st.write(f"Correlación entre {selected_variable_1} y {selected_variable_2}: {round(correlation, 2)}")

            elif (variable_type_1 == "Numérica Continua" or variable_type_1 == "Numérica Discreta") and variable_type_2 == "Tipo Fecha":
                st.subheader("Gráfica de Serie de Tiempo:")
                # Crear una gráfica de serie de tiempo
                df[selected_variable_2] = pd.to_datetime(df[selected_variable_2])
                df.set_index(selected_variable_2, inplace=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(df.index, df[selected_variable_1])
                plt.xlabel("Fecha")
                plt.ylabel(selected_variable_1)
                st.pyplot(fig)

            elif (variable_type_1 == "Numérica Continua" or variable_type_1 == "Numérica Discreta") and variable_type_2 == "Categórica":
                st.subheader("Gráfica de Boxplot:")
                # Crear una gráfica de boxplot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=df, x=selected_variable_2, y=selected_variable_1, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

            elif variable_type_1 == "Categórica" and variable_type_2 == "Categórica":
                st.subheader("Gráfica de Mosaico:")
                # Crear una gráfica de mosaico
                contingency_table = pd.crosstab(df[selected_variable_1], df[selected_variable_2])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                cramer_v = np.sqrt(chi2 / (df.shape[0] * min(contingency_table.shape) - 1))
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
                st.pyplot(fig)

                # Calcular y mostrar el coeficiente de contingencia de Cramer
                st.subheader("Coeficiente de Contingencia de Cramer:")
                st.write(f"Coeficiente de Cramer entre {selected_variable_1} y {selected_variable_2}: {round(cramer_v, 2)}")
