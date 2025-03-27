import pandas as pd
import streamlit as st
from scipy.stats import shapiro, kstest, anderson
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from preprocessamento import preprocessamento

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento
df = preprocessamento(df)

# Colunas para análise
cols = [
    'EPDS_SCORE', 'HADS_SCORE', 'CBTS_SCORE', 'Sleep_hours', 'Age_bb',
    'night_awakening_number_bb1', 'how_falling_asleep_bb1', 'Marital_status_edit',
    'Gestationnal_age', 'Age', 'Education', 'sex_baby1', 'Type_pregnancy'
]

# Função para realizar testes de normalidade
def normality_tests(df, cols):
    results = {}
    summary = []
    for col in cols:
        data = df[col].dropna()
        shapiro_test = shapiro(data)
        ks_test = kstest(data, 'norm', args=(data.mean(), data.std()))
        anderson_test = anderson(data, dist='norm')
        results[col] = {
            'Shapiro-Wilk': shapiro_test,
            'Kolmogorov-Smirnov': ks_test,
            'Anderson-Darling': anderson_test
        }
        # Resumo dos resultados
        is_normal = (shapiro_test.pvalue >= 0.05) and (ks_test.pvalue >= 0.05) and (anderson_test.statistic <= anderson_test.critical_values[2])
        summary.append({'Variable': col, 'Normal': 'SIM' if is_normal else 'NÃO'})
    return results, pd.DataFrame(summary)

# Realizar testes de normalidade
normality_results, summary_df = normality_tests(df, cols)

# Interface Streamlit
st.set_page_config(page_title="Testes de Normalidade", layout="wide")
st.title("Testes de Normalidade")

# Exibir tabela resumida
st.subheader("Resumo dos Testes de Normalidade")
st.dataframe(summary_df)

# Exibir resultados detalhados
for col, tests in normality_results.items():
    st.subheader(f"Resultados para {col}")
    st.write(f"**Shapiro-Wilk**: Statistic={tests['Shapiro-Wilk'][0]:.4f}, p-value={tests['Shapiro-Wilk'][1]:.4f}")
    st.write(f"**Kolmogorov-Smirnov**: Statistic={tests['Kolmogorov-Smirnov'][0]:.4f}, p-value={tests['Kolmogorov-Smirnov'][1]:.4f}")
    st.write(f"**Anderson-Darling**: Statistic={tests['Anderson-Darling'].statistic:.4f}, Critical Values={tests['Anderson-Darling'].critical_values}")

    # Interpretação dos resultados
    st.write("### Interpretação")
    if tests['Shapiro-Wilk'][1] < 0.05:
        st.write("- **Shapiro-Wilk**: Os dados não são normalmente distribuídos (p-valor < 0.05).")
    else:
        st.write("- **Shapiro-Wilk**: Os dados são normalmente distribuídos (p-valor >= 0.05).")

    if tests['Kolmogorov-Smirnov'][1] < 0.05:
        st.write("- **Kolmogorov-Smirnov**: Os dados não são normalmente distribuídos (p-valor < 0.05).")
    else:
        st.write("- **Kolmogorov-Smirnov**: Os dados são normalmente distribuídos (p-valor >= 0.05).")

    if tests['Anderson-Darling'].statistic > tests['Anderson-Darling'].critical_values[2]:
        st.write("- **Anderson-Darling**: Os dados não são normalmente distribuídos (estatística > valor crítico para 5%).")
    else:
        st.write("- **Anderson-Darling**: Os dados são normalmente distribuídos (estatística <= valor crítico para 5%).")

    # Visualizações gráficas
    data = df[col].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(data, kde=True, ax=axes[0])
    axes[0].set_title(f'Histograma de {col}')
    sm.qqplot(data, line='s', ax=axes[1])
    axes[1].set_title(f'Q-Q Plot de {col}')
    st.pyplot(fig)