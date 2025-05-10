import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración inicial
st.set_page_config(layout="wide", page_title="Los Angeles Airbnb Analytics", page_icon="🏨")

# ===== Sistema de Temas =====
css_arena_mar = """
<style>
    .stApp {
        background-color: #F0F2F6;
    }
    section[data-testid="stSidebar"] {
        background-color: #F4E7C5 !important;
        background-image: url('https://www.transparenttextures.com/patterns/beige-paper.png');
        color: #333333 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #333333 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMultiSelect label {
         color: #333333 !important;
    }
     section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
     section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        color: #333333 !important;
     }
    .block-container {
        background: #AFDDFF url('https://www.transparenttextures.com/patterns/white-wave.png') !important;
        background-blend-mode: overlay !important;
        color: #001f3f !important;
        border-radius: 10px;
        padding: 2rem;
    }
     [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: none !important;
        background-color: transparent !important;
     }
    h1, h2, h3, h4, h5, h6 {
        color: #003366;
    }
    .stButton > button {
        background-color: #FF8C00;
        color: white;
        border: none;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #CD6600;
    }
    .stDataFrame {
        background-color: #FFFFFF;
        color: #333333;
    }
</style>
"""

css_oscuro_mejorado = """
<style>
    html, body, .stApp {
        background-color: #121212 !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
        border-right: 1px solid #333333;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stRadio label {
         color: #E0E0E0 !important;
    }
    .block-container {
        background-color: #181818 !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 2rem !important;
        margin-top: 1rem;
        border: 1px solid #2A2A2A;
    }
     div[data-baseweb="select"] > div,
     div[data-baseweb="input"] > input,
     .stTextArea textarea,
     .stDateInput input {
        background-color: #2A2A2A !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
        border-radius: 4px;
     }
     div[data-baseweb="input"] > input::placeholder,
     .stTextArea textarea::placeholder {
         color: #AAAAAA !important;
     }
     .stSlider [data-baseweb="slider"] div[role="slider"]{
         background-color: #0a84ff !important;
     }
      .stSlider label { color: #E0E0E0 !important; }
    .block-container h1, .block-container h2, .block-container h3, .block-container h4, .block-container h5, .block-container h6 {
        color: #00BFFF !important;
    }
     .block-container p, .block-container li, .block-container label, .block-container .stMarkdown {
         color: #FFFFFF !important;
     }
    .stButton > button {
        background-color: #0a84ff;
        color: #FFFFFF !important;
        border: 1px solid #005bb5;
        border-radius: 5px;
        padding: 0.4rem 0.8rem;
    }
    .stButton > button:hover {
        background-color: #005bb5;
        border-color: #0a84ff;
    }
    .stDataFrame {
        background-color: #2A2A2A !important;
        color: #FFFFFF !important;
        border: 1px solid #444444;
        border-radius: 4px;
    }
    .stDataFrame thead th {
        background-color: #333333 !important;
        color: #00BFFF !important;
        border-bottom: 1px solid #555555;
    }
    .stDataFrame tbody td {
         border-color: #444444 !important;
    }
     [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: none !important;
        background-color: #121212 !important;
        border-bottom: 1px solid #333333;
     }
     .plotly-chart {
         background-color: transparent !important;
     }
</style>
"""

# Selector de tema
st.sidebar.title("🎨 Apariencia")
dark_mode = st.sidebar.toggle("Activar Modo Oscuro", value=False)
if dark_mode:
    st.markdown(css_oscuro_mejorado, unsafe_allow_html=True)
else:
    st.markdown(css_arena_mar, unsafe_allow_html=True)

# Cache para carga de datos
@st.cache_resource
def load_data():
    df = pd.read_csv("Datos_limpios_Los_Angeles_Estados_Unidos.csv")
    df = df.drop(['Unnamed: 0'], axis=1)

    # Crear variable binaria explícita basada en el precio
    if 'price' in df.columns:
        avg_price = df['price'].mean()
        df['high_price'] = (df['price'] > avg_price).astype(int)

    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=['latitude', 'longitude'])
    
    numeric_df = df.select_dtypes(include=['float', 'int'])
    text_df = df.select_dtypes(include=['object'])

    # Identificación mejorada de columnas binarias
    binary_cols = []
    for col in numeric_df.columns:
        unique_vals = numeric_df[col].dropna().unique()
        if len(unique_vals) == 2:
            binary_cols.append(col)
    
    unique_room_types = df['room_type'].unique() if 'room_type' in df.columns else []
    
    return df, numeric_df.columns, text_df.columns, unique_room_types, numeric_df, binary_cols

# Cargar datos
df, numeric_cols, text_cols, unique_room_types, numeric_df, binary_cols = load_data()

# ===== Panel de Control en Sidebar =====
st.sidebar.title("🔧 Panel de Control")

# Filtros globales
st.sidebar.markdown("### 🔍 Filtros Globales")
price_min, price_max = st.sidebar.slider("Rango de precios", 
                                       float(df['price'].min()), 
                                       float(df['price'].max()), 
                                       (float(df['price'].min()), float(df['price'].max())))
room_types = st.sidebar.multiselect("Tipos de habitación", 
                                  options=unique_room_types, 
                                  default=unique_room_types)

# Aplicar filtros
filtered_df = df[
    (df['price'] >= price_min) & 
    (df['price'] <= price_max) & 
    (df['room_type'].isin(room_types))
]

# ===== Contenido Principal =====
st.title("🏨 Análisis de Alojamientos Airbnb en Los Ángeles")

# Sección Introductoria
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
    ## 🌴 Los Ángeles, California - La Ciudad de los Ángeles
    
    Los Ángeles es uno de los destinos turísticos más populares del mundo, conocida por:
    - 🌞 Clima soleado durante todo el año
    - 🎬 La meca del cine (Hollywood)
    - 🏖️ Playas icónicas como Santa Mónica y Venice Beach
    - 🎢 Parques temáticos (Disneyland, Universal Studios)
    - 🎨 Vibrante escena artística y cultural
    
    ### 🏡 Mercado Airbnb en Los Ángeles
    El mercado de alquileres vacacionales en LA es uno de los más activos de EE.UU., con:
    - Más de 50,000 propiedades listadas
    - Precios que van desde $50 hasta $10,000 por noche
    - Una gran variedad de tipos de alojamiento
    - Alta demanda durante eventos como los Oscars y el Super Bowl
    """)

with col2:
    try:
        st.image("https://www.rwongphoto.com/images/xl/RWPano066-2_web.jpg", caption="Vista panorámica de Los Ángeles", use_container_width=True)
        st.video("https://www.youtube.com/watch?v=VOM__NZAFQM")
    except:
        st.image("https://images.unsplash.com/photo-1483728642387-6c3bdd6c93e5?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Vista panorámica de Los Ángeles")

# Carrusel visual horizontal con efecto hover
st.markdown("""
<style>
.carusel {
    display: flex;
    width: 100%;
    height: 430px;
    gap: 10px;
}
.carusel img {
    margin-top: 40px;
    width: 0px;
    flex-grow: 1;
    height: 100%;
    object-fit: cover;
    opacity: .8;
    transition: .5s ease;
}
.carusel img:hover {
    cursor: crosshair;
    width: 300px;
    opacity: 1;
    filter: contrast(120%);
}
</style>

<section class="carusel" aria-label="Galería de imágenes de LA.">
    <img src="https://a.travel-assets.com/findyours-php/viewfinder/images/res70/553000/553971-santa-monica-pier.jpg" alt="Santa Monica Beach">
    <img src="https://resizer.glanacion.com/resizer/v2/solo-un-barrio-de-los-angeles-estuvo-excento-de-DHYNHAKAXBC6RD5LMUDUVTKFAQ.jpg?auth=d8d80455db7832854f3c7399e8079190c6e0374ec5fa1f4f2fb0f87276f679dc&width=1280&height=854&quality=70&smart=true" alt="Letrero de Hollywood">
    <img src="https://hips.hearstapps.com/hmg-prod/images/elle-los-angeles01-1559901894.jpg?resize=980:*" alt="Paseo de la fama">
    <img src="https://estaticos-cdn.prensaiberica.es/clip/841aa64b-2796-4ea8-9b2c-5bde40f41590_woman-libre-1200_default_0.jpg" alt="Evento de los Oscars">
</section>
""", unsafe_allow_html=True)

st.markdown('<div style="margin-top: 60px;"></div>', unsafe_allow_html=True)

# Métricas resumidas
st.subheader("📊 Métricas Clave")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Propiedades filtradas", len(filtered_df))
col2.metric("Precio promedio", f"${filtered_df['price'].mean():.2f}")
col3.metric("Evaluaciones promedio", f"{filtered_df['number_of_reviews'].mean():.1f}")
col4.metric("Noches mínimas promedio", f"{filtered_df['minimum_nights'].mean():.1f} noches")

# ===== Vistas Mejoradas =====
def show_views():
    st.sidebar.title("📊 Panel de Control")
    view = st.sidebar.selectbox("Selecciona vista", 
                               ["🏠 Inicio", 
                                "🗺 Mapa Interactivo", 
                                "📈 Gráfico de Líneas", 
                                "🔘 Diagrama de Dispersión", 
                                "🥧 Gráfico de Pastel",
                                "📊 Regresión Lineal Simple",
                                "📊 Regresión Lineal Múltiple", 
                                "📊 Regresión Logística"])

    mostrar_dataset = st.sidebar.checkbox("Mostrar Dataset", key='show_data')
    mostrar_columnas_string = st.sidebar.checkbox("Mostrar columnas tipo texto", key='show_text_cols')

    if mostrar_dataset:
        st.subheader("Dataset completo")
        with st.expander("Ver datos completos"):
            st.write(df)
            st.write("Columnas:", df.columns)
            st.write("Estadísticas descriptivas:", df.describe())

    if mostrar_columnas_string:
        st.subheader("Columnas tipo texto (STRING)")
        st.write(text_cols)   
    
    if view == "🏠 Inicio":
        st.markdown("## 🏡 Bienvenido")
        st.write("Usa la barra lateral para explorar diferentes visualizaciones de los datos de Airbnb en Los Ángeles.")
        
    elif view == "🗺 Mapa Interactivo":
        st.subheader("🗺 Mapa Interactivo de Alojamientos Airbnb en Los Ángeles")

        # Verificar si hay datos después del filtrado
        if len(filtered_df) == 0:
            st.warning("⚠️ No hay propiedades que coincidan con los filtros seleccionados")
            return
    
        # Limitar el muestreo al tamaño del dataframe filtrado
        sample_size = min(800, len(filtered_df))
        sample_df = filtered_df.sample(n=sample_size, random_state=42)
    
        # Calcular el centro del mapa basado en los datos filtrados
        avg_lat = sample_df['latitude'].mean()
        avg_lon = sample_df['longitude'].mean()
    
        # Crear el mapa con ajustes de visualización
        folium_map = folium.Map(
            location=[avg_lat, avg_lon], 
            zoom_start=11,
            control_scale=True
        )
    
        # Agregar cluster de marcadores
        marker_cluster = MarkerCluster().add_to(folium_map)

        # Añadir marcadores para cada propiedad filtrada
        for _, row in sample_df.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"""
                    <b>Tipo:</b> {row['room_type']}<br>
                    <b>Precio:</b> ${row['price']}<br>
                    <b>Evaluaciones:</b> {row['number_of_reviews']}<br>
                    <b>Mínimo noches:</b> {row['minimum_nights']}
                """,
                icon=folium.Icon(
                    color='blue' if row['price'] <= price_max/2 else 'red',
                    icon='home',
                    prefix='fa'
                )
            ).add_to(marker_cluster)
    
        # Mostrar el mapa con controles de tamaño
        st_folium(
            folium_map,
            width=300,
            height=500,
            returned_objects=[],
            use_container_width=True
        )
    
        # Mostrar estadísticas de los filtros aplicados
        st.markdown(f"""
        **Propiedades mostradas:** {len(sample_df)} de {len(filtered_df)} que coinciden con los filtros  
        **Rango de precios:** ${price_min} - ${price_max}  
        **Tipos de habitación:** {', '.join(room_types)}
        """)
    
    elif view == "📈 Gráfico de Líneas":
        st.subheader("📈 Gráfico de Líneas de Los Angeles.")
        
        col1, col2 = st.columns(2)
        with col1:
            variables_lineplot = st.multiselect("Variables numéricas", options=numeric_cols, key='line_vars')
        with col2:
            categoria_lineplot = st.selectbox("Tipo de Habitación", options=unique_room_types, key='room_type')

        if variables_lineplot:
            data = df[df['room_type'] == categoria_lineplot]
            if not data.empty:
                data_features = data[variables_lineplot]
                figure1 = px.line(
                    data_frame=data_features,
                    x=data_features.index,
                    y=variables_lineplot,
                    title='Tendencias por Tipo de Habitación',
                    width=1600, 
                    height=600,
                    color_discrete_sequence=["#261FB3"],
                    template="plotly_dark"
                )
                st.plotly_chart(figure1, use_container_width=True)
            else:
                st.warning("No hay datos disponibles para graficar")
        else:
            st.warning("Selecciona al menos una variable para graficar")

    elif view == "🔘 Diagrama de Dispersión":
        st.subheader("🔘 Gráfico de Dispersión de Los Angeles.")
        
        col1, col2 = st.columns(2)
        with col1:
            x_selected = st.selectbox("Eje X", options=numeric_cols, key='scatter_x')
        with col2:
            y_selected = st.selectbox("Eje Y", options=numeric_cols, key='scatter_y')

        figure2 = px.scatter(
            data_frame=df, 
            x=x_selected, 
            y=y_selected,
            title='Relación entre variables',
            color_discrete_sequence=["#261FB3"],
            template="plotly_dark"
        )
        st.plotly_chart(figure2, use_container_width=True)
    
    elif view == "🥧 Gráfico de Pastel":
        st.subheader("🥧 Gráfico de Pastel de Los Angeles")

        sample_size1 = min(800, len(filtered_df))
        sample_df1 = filtered_df.sample(n=sample_size1, random_state=42)
        
        col1, col2 = st.columns(2)
        with col1:
            var_cat = st.selectbox("Variable Categórica", options=text_cols, key='pie_cat')
        with col2:
            var_num = st.selectbox("Variable Numérica", options=numeric_cols, key='pie_num')

        try:
            figure3 = px.pie(
                data_frame=sample_df1, 
                names=var_cat, 
                values=var_num,
                title='Distribución por Categoría',
                width=1600, 
                height=600,
                color_discrete_sequence=px.colors.sequential.Blues_r,
                template="plotly_dark"
            )
            st.plotly_chart(figure3, use_container_width=True)
        except Exception as e:
            st.error(f"No se puede graficar esta combinación. Error: {str(e)}")
    
    elif view == "📊 Regresión Lineal Simple":
        st.subheader("📊 Regresión Lineal Simple de Los Angeles.")

        if len(numeric_cols) >= 2:
            # Agregamos una opción vacía al principio
            opciones = ["Selecciona una variable"] + list(numeric_cols)

            col1, col2 = st.columns(2)
            with col1:
                X_col = st.selectbox("Variable Independiente (X)", options=opciones, index=0, key='linreg_x')
            with col2:
                y_col = st.selectbox("Variable Dependiente (y)", options=opciones, index=0, key='linreg_y')

            # Solo continuar si ambas variables han sido seleccionadas
            if X_col != "Selecciona una variable" and y_col != "Selecciona una variable":
                temp_df = df[[X_col, y_col]].dropna()
                X = temp_df[[X_col]].values
                y = temp_df[y_col].values

                if len(X) > 0 and len(y) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Coeficientes
                    st.subheader("Coeficientes del Modelo")
                    coef_df = pd.DataFrame({
                        'Componente': ['Coeficiente', 'Intercepto'],
                        'Valor': [model.coef_[0], model.intercept_]
                    })
                    st.dataframe(coef_df, hide_index=True)

                    # R² Score
                    r2_score = model.score(X_test, y_test)
                    st.metric("R² (Coeficiente de Determinación)", f"{r2_score:.4f}")

                    # Gráfico de dispersión + regresión
                    df_plot = pd.DataFrame({
                        X_col: X_test.flatten(),
                        y_col: y_test
                    })

                    fig = px.scatter(
                        df_plot,
                        x=X_col,
                        y=y_col,
                        title="Regresión Lineal Simple",
                        labels={'x': X_col, 'y': y_col},
                        color_discrete_sequence=["#261FB3"],
                        template="plotly_dark"
                    )

                    fig.add_scatter(
                        x=X_test.flatten(), 
                        y=y_pred, 
                        mode='lines', 
                        name='Línea de Regresión',
                        line=dict(color="#261FB3", width=3)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Mapa de calor de correlación
                    st.subheader("Mapa de Calor de Correlación")
                    corr_matrix = df[[X_col, y_col]].corr()

                    fig_heat, ax = plt.subplots()
                    sns.heatmap(corr_matrix, annot=True, cmap="Blues", ax=ax)
                    ax.set_title("Correlación entre Variables")
                    st.pyplot(fig_heat)

                    # Tabla de predicciones
                    st.subheader("Predicciones vs Valores Reales")
                    pred_df = pd.DataFrame({
                        'Real': y_test[:20],
                        'Predicción': y_pred[:20],
                        'Diferencia': abs(y_test[:20] - y_pred[:20])
                    })
                    st.dataframe(pred_df.style.format("{:.2f}"))
            else:
                st.warning("Por favor selecciona ambas variables para continuar.")

    elif view == "📊 Regresión Lineal Múltiple":
        st.subheader("📊 Regresión Lineal Múltiple de Los Angeles.")

        if len(numeric_cols) >= 2:
            X_cols = st.multiselect("Variables Independientes (X)", options=numeric_cols, key='multireg_x')
            y_col = st.selectbox("Variable Dependiente (y)", options=numeric_cols, key='multireg_y')
        
            if len(X_cols) >= 1 and y_col:
                temp_df = df[X_cols + [y_col]].dropna()
                X = temp_df[X_cols].values
                y = temp_df[y_col].values
        
                if len(X) > 0 and len(y) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            
                    # Coeficientes del modelo
                    st.subheader("Coeficientes del Modelo")
                    coef_df = pd.DataFrame({
                        'Variable': X_cols + ['Intercepto'],
                        'Coeficiente': list(model.coef_) + [model.intercept_]
                    })
                    st.dataframe(coef_df, hide_index=True)

                    # ✅ NUEVO: R² Score
                    st.metric("R² (Coeficiente de Determinación)", f"{model.score(X_test, y_test):.4f}")

                    # Gráfico de dispersión
                    fig = px.scatter(
                        x=y_test, 
                        y=y_pred, 
                        title="Valores Reales vs Predicciones",
                        labels={'x': 'Valores Reales', 'y': 'Predicciones'},
                        color_discrete_sequence=["#261FB3"],
                        template="plotly_dark"
                    )
                    fig.add_shape(
                        type="line", 
                        x0=min(y_test), 
                        y0=min(y_test),
                        x1=max(y_test), 
                        y1=max(y_test),
                        line=dict(color="#261FB3", dash="dash", width=2)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ✅ NUEVO: Mapa de calor de correlación
                    st.subheader("Mapa de Calor de Correlación")
                    corr_matrix = temp_df.corr()
                    fig_heat, ax = plt.subplots()
                    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
                    ax.set_title("Correlación entre Variables")
                    st.pyplot(fig_heat)

                    # Tabla de predicciones
                    st.subheader("Predicciones vs Valores Reales")
                    pred_df = pd.DataFrame({
                        'Real': y_test[:20],
                        'Predicción': y_pred[:20],
                        'Diferencia': abs(y_test[:20] - y_pred[:20])
                    })
                    st.dataframe(pred_df.style.format("{:.2f}"))

    elif view == "📊 Regresión Logística":
        st.subheader("📊 Regresión Logística de Los Angeles.")

        if len(numeric_cols) >= 1 and len(binary_cols) >= 1:
            st.write("*Variables disponibles para clasificación:*", binary_cols)
        
            X_cols = st.multiselect(
                "Variables Independientes (X)", 
                options=[col for col in numeric_cols if col not in binary_cols],
                key='logreg_x'
            )
            y_col = st.selectbox(
                "Variable Dependiente (y - binaria)", 
                options=binary_cols,
                key='logreg_y'
            )
        
            if len(X_cols) >= 1:
                temp_df = df[X_cols + [y_col]].dropna()
                X = temp_df[X_cols].values
                y = temp_df[y_col].values
            
                if len(X) > 0 and len(y) > 0:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                    # Gráfico de coeficientes
                    fig = px.bar(
                        x=X_cols, 
                        y=model.coef_[0], 
                        title="Importancia de Variables",
                        labels={'x': 'Variables', 'y': 'Coeficientes'},
                        color_discrete_sequence=["#261FB3"],
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                    # Matriz de confusión
                    st.subheader("Matriz de Confusión")
                    conf_matrix = pd.crosstab(
                        pd.Series(y_test, name='Real'), 
                        pd.Series(y_pred, name='Predicción')
                    )
                    st.write(conf_matrix)

# Ejecutar la aplicación
show_views()
