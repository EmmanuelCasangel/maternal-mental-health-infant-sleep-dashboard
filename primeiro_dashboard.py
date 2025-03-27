# importar bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# fazer a leitura do arquivo
caminho_arquivo = 'C:/Users/Emille/Documents/UNIFESP/MATÉRIAS/Tópicos em Ciência de Dados para Neurociência/maternal health.xlsx'
df = pd.read_excel(caminho_arquivo)

# 2. Criar a coluna de score HADS
hads_columns = ['HADS_1', 'HADS_3', 'HADS_5', 'HADS_7', 'HADS_9', 'HADS_11', 'HADS_13']
df['HADS_Score'] = df[hads_columns].sum(axis=1)

# 3. Criar a coluna de categorias do score HADS
bins = [1, 8, 12, 22]
labels = ['improvável', 'possível', 'provável']
df['HADS_Category'] = pd.cut(df['HADS_Score'], bins=bins, labels=labels, right=False)
df['HADS_Category_numeric'] = df['HADS_Category'].astype('category').cat.codes

# 4. Correlação de Spearman
correlation, p_value = spearmanr(df['HADS_Score'], df['Education'])
correlation_category, p_value_category = spearmanr(df['HADS_Category_numeric'], df['Education'])

# 5. Dashboard Streamlit
st.title("Impact of Maternal Mental Health on Infant Sleep")

# Seção: Distribuição de Idade
st.header("Distribution of maternal age by age group")

# Gráfico de pizza com faixas etárias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ... (seu código de carregamento de dados e criação de categorias de idade)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ... (seu código de carregamento de dados e criação de categorias de idade)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ... (seu código de carregamento de dados e criação de categorias de idade)

# Gráfico de pizza com faixas etárias, porcentagens e legenda personalizada
bins_age = [19, 26, 31, 36, 48]
labels_age = ['19-25', '26-30', '31-35', '36-47']
df['Age_Category'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)
age_counts = df['Age_Category'].value_counts()
fig1, ax1 = plt.subplots()

# Paleta de cores personalizada
cores_grafico = ['#2d2e3f', '#ce3450', '#e2d6ca', '#99b8c1', '#e6ba3d', '#91c7bc']

# Crie o gráfico de pizza com porcentagens nas fatias
patches, texts, autotexts = ax1.pie(age_counts, autopct='%1.1f%%', startangle=140, colors=cores_grafico)
ax1.axis('equal')

# Crie a legenda personalizada
plt.legend(patches, age_counts.index, loc="best")

st.pyplot(fig1)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ... (seu código de carregamento de dados)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ... (seu código de carregamento de dados)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ... (seu código de carregamento de dados)

# Gráfico de barras com cores e legendas personalizadas
education_counts = df['Education'].value_counts().sort_index()

fig2, ax2 = plt.subplots(figsize=(12, 6))

# Paleta de cores personalizada
cores_grafico = ['#2d2e3f', '#e2d6ca', '#99b8c1', '#e6ba3d', '#91c7bc']

# Mapear os valores numéricos para os rótulos descritivos
education_labels = {
    1: 'no education',
    2: 'compulsory school',
    3: 'post-compulsory education',
    4: 'university of Applied Science or University Technology Degree',
    5: 'university'
}

# Criar barras com cores e rótulos individuais
bars = ax2.bar(education_counts.index, education_counts.values, color=cores_grafico)

# Adicionar rótulos de legenda para cada barra
for bar, label in zip(bars, education_counts.index):
    bar.set_label(education_labels[label])

ax2.set_xticks(education_counts.index)
ax2.set_xticklabels(education_counts.index, rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

ax2.set_xlabel('Nível de Escolaridade')
ax2.set_ylabel('Frequência')
ax2.set_title('Distribuição do Nível de Escolaridade')

# Criar a legenda
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
st.pyplot(fig2)

# Seção: Correlação Spearman
st.header("Correlação Spearman")

# Tabela de correlação
st.subheader("Correlação HADS_Score e Education")
correlation_table = pd.DataFrame({
    'Correlação Spearman': [correlation],
    'Valor p': [p_value]
})
st.dataframe(correlation_table)

st.subheader("Correlação HADS_Category e Education")
correlation_category_table = pd.DataFrame({
    'Correlação Spearman (Categoria)': [correlation_category],
    'Valor p (Categoria)': [p_value_category]
})
st.dataframe(correlation_category_table)

# Mapa de calor
st.subheader("Mapa de Calor da Correlação")
fig3, ax3 = plt.subplots()
sns.heatmap(df[['HADS_Score', 'Education']].corr(method='spearman'), annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

st.subheader("Mapa de Calor da Correlação (Categoria)")
fig4, ax4 = plt.subplots()
sns.heatmap(df[['HADS_Category_numeric', 'Education']].corr(method='spearman'), annot=True, cmap='coolwarm', ax=ax4)
st.pyplot(fig4)