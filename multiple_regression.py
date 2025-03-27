import pandas as pd
import statsmodels.api as sm
import streamlit as st
from preprocessamento import preprocessamento

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento
df = preprocessamento(df)

# Interface Streamlit
st.set_page_config(page_title="Análise de Regressão Linear Múltipla", layout="wide")
st.title("Análise de Regressão Linear Múltipla")

# Descrição da análise
st.markdown("""
Esta aplicação realiza uma análise de regressão linear múltipla para entender a relação entre diferentes variáveis e a pontuação EPDS.
Selecione as variáveis independentes e a variável dependente para visualizar os resultados.
""")

# Selecionar variáveis independentes e dependente
all_columns = df.columns.tolist()
dependent_var = st.selectbox('Selecione a variável dependente', all_columns, index=all_columns.index('EPDS_SCORE'))
independent_vars = st.multiselect('Selecione as variáveis independentes', all_columns, default=['Sleep_hours', 'night_awakening_number_bb1', 'Marital_status_edit'])

if independent_vars:
    X = df[independent_vars]
    y = df[dependent_var]

    # Adicionar constante para o modelo
    X = sm.add_constant(X)

    # Ajustar o modelo de regressão linear múltipla
    model = sm.OLS(y, X).fit()

    # Exibir resumo do modelo
    st.subheader("Resumo do Modelo")
    model_summary = model.summary().as_text()
    st.text_area("Resumo do Modelo", model_summary, height=400)

    # Adicionar dicionário explicativo
    st.subheader("Interpretação dos Resultados")
    st.markdown("""
    **Dicionário de Saídas:**

    - **Dep. Variable**: A variável dependente que está sendo prevista pelo modelo.
    - **R-squared**: A proporção da variância na variável dependente que é explicada pelas variáveis independentes. Valores mais próximos de 1 indicam um melhor ajuste.
    - **Adj. R-squared**: O R-squared ajustado para o número de variáveis no modelo. É uma medida mais precisa do ajuste do modelo.
    - **F-statistic**: Teste F para a significância global do modelo. Valores altos indicam que pelo menos uma variável independente é significativa.
    - **Prob (F-statistic)**: O valor p associado ao teste F. Valores menores que 0.05 indicam que o modelo é estatisticamente significativo.
    - **Log-Likelihood**: A log-verossimilhança do modelo. Valores mais altos indicam um melhor ajuste.
    - **AIC/BIC**: Critérios de informação de Akaike e Bayesiano. Valores menores indicam um modelo melhor.
    - **Df Residuals**: O número de graus de liberdade dos resíduos.
    - **Df Model**: O número de graus de liberdade do modelo.
    - **coef**: Os coeficientes estimados para cada variável independente. Indicam a mudança esperada na variável dependente para uma unidade de mudança na variável independente.
    - **std err**: O erro padrão dos coeficientes. Medida da precisão das estimativas dos coeficientes.
    - **t**: O valor t para o teste de significância dos coeficientes.
    - **P>|t|**: O valor p associado ao teste t. Valores menores que 0.05 indicam que o coeficiente é estatisticamente significativo.
    - **[0.025, 0.975]**: O intervalo de confiança de 95% para os coeficientes.
    - **Omnibus**: Teste de normalidade dos resíduos. Valores altos indicam que os resíduos não são normalmente distribuídos.
    - **Prob(Omnibus)**: O valor p associado ao teste Omnibus. Valores menores que 0.05 indicam que os resíduos não são normalmente distribuídos.
    - **Durbin-Watson**: Teste de autocorrelação dos resíduos. Valores próximos de 2 indicam ausência de autocorrelação.
    - **Jarque-Bera (JB)**: Teste de normalidade dos resíduos. Valores altos indicam que os resíduos não são normalmente distribuídos.
    - **Prob(JB)**: O valor p associado ao teste Jarque-Bera. Valores menores que 0.05 indicam que os resíduos não são normalmente distribuídos.
    - **Skew**: A assimetria dos resíduos. Valores diferentes de 0 indicam assimetria.
    - **Kurtosis**: A curtose dos resíduos. Valores diferentes de 3 indicam curtose anormal.
    - **Cond. No.**: O número de condição do modelo. Valores altos indicam problemas de multicolinearidade.
    """)

else:
    st.write("Por favor, selecione pelo menos uma variável independente.")