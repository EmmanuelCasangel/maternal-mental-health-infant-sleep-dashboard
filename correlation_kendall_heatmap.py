import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from preprocessamento import preprocessamento

def calculate_kendall_correlations(df, cols):
    correlations = pd.DataFrame(index=cols, columns=cols)
    p_values = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            corr, p = kendalltau(df[col1], df[col2])
            correlations.loc[col1, col2] = corr
            p_values.loc[col1, col2] = p
    return correlations, p_values

def plot_heatmap(correlations):
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations.astype(float), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.1)
    plt.title('Mapa de Correlação (Kendall)')
    st.pyplot(plt)

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento
df = preprocessamento(df)

# Interface Streamlit
st.set_page_config(page_title="Mapa de Correlação", layout="wide")

# Calcular e plotar heatmap de correlações
st.title("Mapa de Correlação (Kendall)")
cols = [
    'EPDS_SCORE', 'HADS_SCORE', 'CBTS_SCORE', 'Sleep_hours', 'Age_bb',
    'night_awakening_number_bb1', 'how_falling_asleep_bb1', 'Marital_status_edit',
    'Gestationnal_age', 'Age', 'Education', 'sex_baby1', 'Type_pregnancy'
]
correlations, p_values = calculate_kendall_correlations(df, cols)
plot_heatmap(correlations)