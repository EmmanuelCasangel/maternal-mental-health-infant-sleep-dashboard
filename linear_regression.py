import pandas as pd
import statsmodels.api as sm
from preprocessamento import preprocessamento

# Carregar dados
df = pd.read_csv('Dataset_maternal_mental_health_infant_sleep.csv', encoding='latin1')

# Pré-processamento
df = preprocessamento(df)

# Selecionar variáveis
X = df['EPDS_SCORE']
y = df['Sleep_hours']

# Adicionar constante para o modelo
X = sm.add_constant(X)

# Ajustar o modelo de regressão linear
model = sm.OLS(y, X).fit()

# Resumo do modelo
print(model.summary())