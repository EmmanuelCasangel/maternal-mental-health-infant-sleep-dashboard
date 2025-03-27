import pandas as pd
import streamlit as st
import plotly.express as px
from preprocessamento import preprocessamento, traduzir_valores

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento corrigido
df = preprocessamento(df)
df = traduzir_valores(df)

# Interface Streamlit
st.set_page_config(page_title="Dashboard Saúde Materno-Infantil", layout="wide")

# Sidebar
st.sidebar.header("Filtros")
age_range = st.sidebar.slider("Idade da Mãe",
                              min_value=int(df['Age'].min()),
                              max_value=int(df['Age'].max()),
                              value=(int(df['Age'].min()), int(df['Age'].max())),
                              step=1)

selected_education = st.sidebar.multiselect("Escolaridade",
                                            options=df['Education'].unique(),
                                            default=df['Education'].unique())

# Aplicar filtros
filtered_df = df[
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Education'].isin(selected_education))
]

# Visualizações principais
st.title("Análise de Saúde Mental Materna e Sono Infantil")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Média EPDS", f"{filtered_df['EPDS_SCORE'].mean():.1f}")
with col2:
    st.metric("Média HADS", f"{filtered_df['HADS_SCORE'].mean():.1f}")
with col3:
    st.metric("Horas de Sono Médio", f"{filtered_df['Sleep_hours'].mean():.1f}")

# Gráficos
st.subheader("")
tab1, tab2, tab3 = st.tabs(["Analise Descritiva", "Relação entre saude mental e sono", "analises e ferramenta"])

with tab1:
    # Visualização adicional para HADS_Category
    st.title("Distribuição das Categorias HADS")

    # Gráfico de barras para categorias HADS
    hads_category_counts = df['HADS_Category'].value_counts().reset_index()
    hads_category_counts.columns = ['HADS_Category', 'count']

    fig = px.bar(hads_category_counts,
                 x='HADS_Category', y='count',
                 labels={'HADS_Category': 'Categoria', 'count': 'Contagem'},
                 title='Distribuição das Categorias HADS')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.density_contour(filtered_df,
                             x='HADS_SCORE',
                             y='CBTS_SCORE',
                             color='Marital_status')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_df,
                           x='EPDS_SCORE',
                           nbins=20,
                           color='sex_baby1')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(filtered_df,
                     y='Sleep_hours',
                     x='Marital_status')
        st.plotly_chart(fig, use_container_width=True)