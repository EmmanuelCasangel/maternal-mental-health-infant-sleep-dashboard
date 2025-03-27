import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

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
    sns.set(style="whitegrid")  # Define a consistent style
    sns.heatmap(correlations.astype(float), cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.1)
    plt.title('Mapa de Correlação (Kendall)')
    st.pyplot(plt)