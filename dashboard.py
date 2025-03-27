import pandas as pd
import streamlit as st
import plotly.express as px
from preprocessamento import preprocessamento

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento corrigido
df = preprocessamento(df)

# Interface Streamlit
st.set_page_config(page_title="Dashboard Saúde Materno-Infantil", layout="wide")

# Exibir dados processados
st.subheader("Dados Processados")
st.dataframe(df, use_container_width=True)

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
st.subheader("Relação entre Variáveis")
tab1, tab2, tab3 = st.tabs(["EPDS vs Sono", "HADS vs CBTS", "Distribuições"])

with tab1:
    fig = px.scatter(filtered_df,
                     x='EPDS_SCORE',
                     y='Sleep_hours',
                     trendline="ols")
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

# Análise de dados brutos
st.subheader("Dados Brutos")
st.dataframe(filtered_df[[
    'Participant_number', 'Age', 'Marital_status', 'Education',
    'EPDS_SCORE', 'HADS_SCORE', 'CBTS_SCORE', 'Sleep_hours'
]].head(10), use_container_width=True)