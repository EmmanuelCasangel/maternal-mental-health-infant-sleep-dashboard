import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from preprocessamento import preprocessamento

def calculate_correlations(df, cols):
    correlations = pd.DataFrame(index=cols, columns=cols)
    p_values = pd.DataFrame(index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            corr, p = spearmanr(df[col1], df[col2])
            correlations.loc[col1, col2] = corr
            p_values.loc[col1, col2] = p
    return correlations, p_values

def plot_heatmap(correlations):
    plt.figure(figsize=(4, 3))  # Further reduce the size of the figure
    sns.heatmap(correlations.astype(float), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.1)  # Adjust the linewidths
    plt.title('Mapa de Correlação (Spearman)')
    st.pyplot(plt)

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento
df = preprocessamento(df)

# Interface Streamlit
st.set_page_config(page_title="Mapa de Correlação", layout="wide")

# Calcular e plotar heatmap de correlações
st.title("Mapa de Correlação (Spearman)")
cols = ['EPDS_SCORE', 'HADS_SCORE', 'CBTS_SCORE', 'Sleep_hours']
correlations, p_values = calculate_correlations(df, cols)
plot_heatmap(correlations)