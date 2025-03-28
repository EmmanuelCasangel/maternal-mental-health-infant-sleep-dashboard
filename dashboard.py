import pandas as pd
import streamlit as st
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np  # Importing numpy

from preprocessamento import preprocessamento, traduzir_valores
from correlation_kendall_heatmap import calculate_kendall_correlations, plot_heatmap as plot_kendall_heatmap
from correlation_spearman_heatmap import calculate_correlations, plot_heatmap as plot_spearman_heatmap
from multiple_regression import perform_multiple_regression
from teste_normalidade import normality_tests

# Interface Streamlit
st.set_page_config(page_title="Dashboard Saúde Materno-Infantil", layout="wide")

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento corrigido
df = preprocessamento(df)
df_translate = traduzir_valores(df.copy())

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

# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Média EPDS", f"{filtered_df_translate['EPDS_SCORE'].mean():.1f}")
# with col2:
#     st.metric("Média HADS", f"{filtered_df_translate['HADS_SCORE'].mean():.1f}")
# with col3:
#     st.metric("Horas de Sono Médio", f"{filtered_df_translate['Sleep_hours'].mean():.1f}")

# Gráficos
st.subheader("")
tab1, tab2, tab3, tab4 = st.tabs(["Analise Descritiva", "Fatores Associados à Depressão Pós-Parto (EPDS_SCORE)", "analises e ferramenta", "testes normalidade"])

with tab1:
    st.header("Distribution of maternal age")

    # Gráfico de pizza com faixas etárias, porcentagens e legenda personalizada
    bins_age = [19, 26, 31, 36, 48]
    labels_age = ['19-25', '26-30', '31-35', '36-47']
    df_translate['Age_Category'] = pd.cut(df_translate['Age'], bins=bins_age, labels=labels_age, right=False)
    age_counts = df_translate['Age_Category'].value_counts()
    fig1, ax1 = plt.subplots()

    # Paleta de cores personalizada
    cores_grafico = ['#2d2e3f', '#ce3450', '#e2d6ca', '#99b8c1', '#e6ba3d', '#91c7bc']

    # Crie o gráfico de pizza com porcentagens nas fatias
    patches, texts, autotexts = ax1.pie(age_counts, autopct='%1.1f%%', startangle=140, colors=cores_grafico)
    ax1.axis('equal')

    # Crie a legenda personalizada
    plt.legend(patches, age_counts.index, loc="best")

    st.pyplot(fig1)

    # Seção: Distribuição do Nível de Escolaridade (Título com o mesmo tamanho)
    st.header("Distribuição do Nível de Escolaridade")

    # Gráfico de barras com cores e legendas personalizadas
    education_counts = df_translate['Education'].value_counts().sort_index()

    # Aumentar o tamanho da figura
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    # Criar barras com cores e rótulos individuais
    bars = ax2.bar(education_counts.index, education_counts.values, color=cores_grafico)

    ax2.set_xticks(education_counts.index)
    ax2.set_xticklabels(education_counts.index, rotation=0, ha='center')

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    ax2.set_xlabel('Education')
    ax2.set_ylabel('Frequency')

    # Criar a legenda dentro do gráfico
    ax2.legend(loc='upper left')

    plt.tight_layout()
    st.pyplot(fig2)
########################################################################################################################
    # Section: Marital status
    st.header("Distribution of marital status")

    # Count the frequency of each marital status
    marital_status_counts = df['Marital_status_edit'].value_counts()

    # Dictionary mapping numeric values to descriptive labels
    marital_status_labels = {
        1: '1 = Single',
        2: '2 = In a relationship',
        3: '3 = Separated, divorced, or widowed',
        6: '6 = Other'
    }

    # Custom colors for the bars
    cores_barras = ['#ce3450', '#e2d6ca', '#99b8c1', '#e6ba3d']  # Adjust colors as needed

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(marital_status_counts.index, marital_status_counts.values, color=cores_barras)

    # Add legend labels for each bar
    for bar, label in zip(bars, marital_status_counts.index):
        bar.set_label(marital_status_labels[label])

    # Add labels and title
    ax.set_xlabel('Marital status')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=0, ha='right')

    # Add frequency numbers above each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')

    # Create the legend inside the chart
    ax.legend(loc='upper right')  # Adjust legend location as needed

    plt.tight_layout()

    # Integrate with Streamlit
    st.pyplot(fig)

    # Distribuição da Idade do Bebê
    st.header("Baby Age Distribution")

    # Mapear os valores para os rótulos desejados
    age_categories = {
        1: '≥3 months to <6 months',
        2: '≥6 months to <9 months',
        3: '≥9 months to <12 months'
    }

    # Contar a frequência das categorias
    category_counts = df_translate['Age_bb'].value_counts().sort_index()

    # Mapear os índices para os rótulos
    category_counts.index = category_counts.index.map(age_categories)

    # Criar o gráfico de barras
    fig_age, ax_age = plt.subplots()
    bars = ax_age.bar(category_counts.index, category_counts.values, color=cores_grafico)

    # Adicionar a frequência em cima de cada barra
    for bar in bars:
        yval = bar.get_height()
        ax_age.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    # Configurar rótulos dos eixos
    ax_age.set_xlabel("Baby Age Intervals")
    ax_age.set_ylabel("Frequency")

    # Configurar a rotação dos rótulos do eixo x
    plt.xticks(rotation=45, ha='right')

    # Criar legenda no lado direito inferior
    ax_age.legend(bars, category_counts.index, title="Baby Age Intervals", loc='lower right')

    # Exibir o gráfico no Streamlit
    st.pyplot(fig_age)

    # Número de Despertares Noturnos
    st.header("Night Awakenings Frequency")

    # Contar a frequência dos despertares noturnos
    awake_counts = df_translate['night_awakening_number_bb1'].value_counts().sort_index()

    # Criar o gráfico de barras
    fig_awake, ax_awake = plt.subplots()
    bars = ax_awake.bar(awake_counts.index, awake_counts.values, color=cores_grafico)

    # Adicionar a frequência em cima de cada barra
    for bar in bars:
        yval = bar.get_height()
        ax_awake.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    # Configurar rótulos dos eixos
    ax_awake.set_xlabel("Number of Night Awakenings")
    ax_awake.set_ylabel("Frequency")

    # Remover a legenda
    # ax_awake.legend(["Night Awakenings"], loc="best") # Removido

    # Exibir o gráfico no Streamlit
    st.pyplot(fig_awake)

    # Qualidade do Sono do Bebê
    st.header("Infant Sleep Quality Categories (Fixed Frequencies)")

    # Frequências corretas
    category_frequencies = {
        'Fed': 90,
        'Rocked': 74,
        'Held': 22,
        'Alone': 177,
        'Parental': 74
    }

    # Criar DataFrame com as frequências e ordenar do maior para o menor
    df_frequencies = pd.Series(category_frequencies).sort_values(ascending=False)

    # Definir cores para cada barra
    cores_barras = ['#2d2e3f', '#ce3450', '#e2d6ca', '#99b8c1',
                    '#e6ba3d']  # Cores para Fed, Alone, Rocked, Parental, Held

    # Criar o gráfico de barras
    fig_sleep, ax_sleep = plt.subplots()
    bars = ax_sleep.bar(df_frequencies.index, df_frequencies.values, color=cores_barras)

    # Adicionar a frequência em cima de cada barra
    for bar in bars:
        yval = bar.get_height()
        ax_sleep.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom')

    # Configurar rótulos dos eixos
    ax_sleep.set_xlabel("Sleep Categories")
    ax_sleep.set_ylabel("Frequency")

    # Criar legenda com rótulos ajustados
    legenda_labels = {
        'Alone': 'Alone in the crib',
        'Fed': 'While being fed',
        'Rocked': 'While being rocked',
        'Parental': 'In the crib with parental presence',
        'Held': 'While being held'
    }

    ax_sleep.legend(bars, [legenda_labels[label] for label in df_frequencies.index], title="Sleep Categories")

    # Exibir o gráfico no Streamlit
    st.pyplot(fig_sleep)
########################################################################################################################
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
    st.title("Fatores Associados à Depressão Pós-Parto (EPDS_SCORE)")

    # Gráficos de caixa interativos com linha de tendência
    st.subheader("Relação entre EPDS_SCORE e Variáveis Independentes")

    # Sleep_hours vs EPDS_SCORE
    fig1 = px.box(filtered_df_translate, x='Sleep_hours', y='EPDS_SCORE',
                  title='EPDS_SCORE vs Sleep_hours',
                  labels={'Sleep_hours': 'Horas de Sono', 'EPDS_SCORE': 'EPDS_SCORE'})
    x_vals = filtered_df_translate['Sleep_hours']
    y_vals = filtered_df_translate['EPDS_SCORE']
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    fig1.add_traces(px.line(x=x_vals, y=p(x_vals), labels={'x': 'Horas de Sono', 'y': 'EPDS_SCORE'}).data)
    st.plotly_chart(fig1, use_container_width=True)

    # night_awakening_number_bb1 vs EPDS_SCORE
    fig2 = px.box(filtered_df_translate, x='night_awakening_number_bb1', y='EPDS_SCORE',
                  title='EPDS_SCORE vs Número de Despertares Noturnos',
                  labels={'night_awakening_number_bb1': 'Número de Despertares Noturnos', 'EPDS_SCORE': 'EPDS_SCORE'})
    x_vals = filtered_df_translate['night_awakening_number_bb1']
    y_vals = filtered_df_translate['EPDS_SCORE']
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    fig2.add_traces(px.line(x=x_vals, y=p(x_vals), labels={'x': 'Número de Despertares Noturnos', 'y': 'EPDS_SCORE'}).data)
    st.plotly_chart(fig2, use_container_width=True)

    # Education vs EPDS_SCORE (simple box plot)
    fig3 = px.box(filtered_df_translate, x='Education', y='EPDS_SCORE',
                  title='EPDS_SCORE vs Educação',
                  labels={'Education': 'Educação', 'EPDS_SCORE': 'EPDS_SCORE'})
    st.plotly_chart(fig3, use_container_width=True)

    # Explicações sobre as relações observadas
    st.subheader("Interpretação das Relações")
    st.markdown("""
    - **Sleep_hours**: Observa-se uma relação negativa entre as horas de sono e a pontuação EPDS, indicando que mais horas de sono estão associadas a menores pontuações de depressão pós-parto.
    - **night_awakening_number_bb1**: Há uma tendência de aumento na pontuação EPDS com o aumento do número de despertares noturnos, sugerindo que mais despertares noturnos podem estar associados a maiores níveis de depressão pós-parto.
    - **Education**: Níveis mais altos de educação parecem estar associados a menores pontuações de EPDS, indicando que a educação pode ter um efeito protetor contra a depressão pós-parto.
    """)

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
    perform_multiple_regression(df)

with tab4:
    st.title("Testes de Normalidade")

    # Realizar testes de normalidade
    normality_results, summary_df = normality_tests(df, cols)

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