import pandas as pd
import streamlit as st
import plotly.express as px

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')


# Pré-processamento corrigido
def preprocessamento(df):
    # Converter variáveis categóricas
    marital_status_map = {
        1: 'Solteira',
        2: 'Casada/União estável',
        3: 'Outro'
    }
    education_map = {
        1: 'Fundamental',
        2: 'Médio',
        3: 'Superior',
        4: 'Pós-graduação',
        5: 'Outro'
    }

    df['Marital_status'] = df['Marital_status'].map(marital_status_map)
    df['Education'] = df['Education'].map(education_map)
    df['sex_baby1'] = df['sex_baby1'].map({1: 'Masculino', 2: 'Feminino'})

    # Calcular escores
    df['EPDS_Total'] = df[[f'EPDS_{i}' for i in range(1, 11)]].sum(axis=1)
    df['HADS_Total'] = df[['HADS_1', 'HADS_3', 'HADS_5', 'HADS_7', 'HADS_9', 'HADS_11', 'HADS_13']].sum(axis=1)

    # Corrigir colunas CBTS (a nomenclatura muda após o 12)
    cbts_columns = [f'CBTS_M_{i}' for i in range(3, 13)] + [f'CBTS_{i}' for i in range(13, 23)]
    df['CBTS_Total'] = df[cbts_columns].sum(axis=1)

    # Converter duração do sono
    def convert_sleep_duration(time_str):
        try:
            if pd.isna(time_str):
                return None
            hours, minutes = map(int, str(time_str).split(':'))
            return hours + minutes / 60
        except:
            return None

    df['Sleep_hours'] = df['Sleep_night_duration_bb1'].apply(convert_sleep_duration)

    return df


df = preprocessamento(df)

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
    st.metric("Média EPDS", f"{filtered_df['EPDS_Total'].mean():.1f}")
with col2:
    st.metric("Média HADS", f"{filtered_df['HADS_Total'].mean():.1f}")
with col3:
    st.metric("Horas de Sono Médio", f"{filtered_df['Sleep_hours'].mean():.1f}")

# Gráficos
st.subheader("Relação entre Variáveis")
tab1, tab2, tab3 = st.tabs(["EPDS vs Sono", "HADS vs CBTS", "Distribuições"])

with tab1:
    fig = px.scatter(filtered_df,
                     x='EPDS_Total',
                     y='Sleep_hours',
                     color='Education',
                     trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.density_contour(filtered_df,
                             x='HADS_Total',
                             y='CBTS_Total',
                             color='Marital_status')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_df,
                           x='EPDS_Total',
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
    'EPDS_Total', 'HADS_Total', 'CBTS_Total', 'Sleep_hours'
]].head(10), use_container_width=True)