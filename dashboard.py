import pandas as pd
import streamlit as st
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from preprocessamento import preprocessamento, traduzir_valores
from correlation_kendall_heatmap import calculate_kendall_correlations, plot_heatmap as plot_kendall_heatmap
from correlation_spearman_heatmap import calculate_correlations, plot_heatmap as plot_spearman_heatmap
from multiple_regression import perform_multiple_regression

# Interface Streamlit
st.set_page_config(page_title="Dashboard Saúde Materno-Infantil", layout="wide")

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento corrigido
df = preprocessamento(df)
df_translate = traduzir_valores(df)

# Sidebar
st.sidebar.header("Filtros")
age_range = st.sidebar.slider("Idade da Mãe",
                              min_value=int(df_translate['Age'].min()),
                              max_value=int(df_translate['Age'].max()),
                              value=(int(df_translate['Age'].min()), int(df_translate['Age'].max())),
                              step=1)

selected_education = st.sidebar.multiselect("Escolaridade",
                                            options=df_translate['Education'].unique(),
                                            default=df_translate['Education'].unique())

# Aplicar filtros
filtered_df_translate = df_translate[
    (df_translate['Age'].between(age_range[0], age_range[1])) &
    (df_translate['Education'].isin(selected_education))
]

# Visualizações principais
st.title("Análise de Saúde Mental Materna e Sono Infantil")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Média EPDS", f"{filtered_df_translate['EPDS_SCORE'].mean():.1f}")
with col2:
    st.metric("Média HADS", f"{filtered_df_translate['HADS_SCORE'].mean():.1f}")
with col3:
    st.metric("Horas de Sono Médio", f"{filtered_df_translate['Sleep_hours'].mean():.1f}")

# Gráficos
st.subheader("")
tab1, tab2, tab3, tab4 = st.tabs(["Analise Descritiva", "Relação entre saude mental e sono", "analises e ferramenta", "testes normalidade"])

with tab1:
    # Visualização adicional para HADS_Category
    st.title("Distribuição das Categorias HADS")

    # Gráfico de barras para categorias HADS
    hads_category_counts = df_translate['HADS_Category'].value_counts().reset_index()
    hads_category_counts.columns = ['HADS_Category', 'count']

    fig = px.bar(hads_category_counts,
                 x='HADS_Category', y='count',
                 labels={'HADS_Category': 'Categoria', 'count': 'Contagem'},
                 title='Distribuição das Categorias HADS')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.density_contour(filtered_df_translate,
                             x='HADS_SCORE',
                             y='CBTS_SCORE',
                             color='Marital_status')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    cols = [
        'EPDS_SCORE', 'HADS_SCORE', 'CBTS_SCORE', 'Sleep_hours', 'Age_bb',
        'night_awakening_number_bb1', 'how_falling_asleep_bb1', 'Marital_status_edit',
        'Gestationnal_age', 'Age', 'Education', 'sex_baby1', 'Type_pregnancy'
    ]

    kendall_correlations, _ = calculate_kendall_correlations(filtered_df_translate, cols)
    spearman_correlations, _ = calculate_correlations(filtered_df_translate, cols)

    tab3_col1, tab3_col2 = st.columns(2)
    with tab3_col1:
        st.subheader("Mapa de Correlação (Kendall)")
        plot_kendall_heatmap(kendall_correlations)

    with tab3_col2:
        st.subheader("Mapa de Correlação (Spearman)")
        plot_spearman_heatmap(spearman_correlations)

    st.subheader("Análise de Regressão Linear Múltipla")
    perform_multiple_regression(filtered_df_translate)
