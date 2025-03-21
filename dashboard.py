import streamlit as st  # Por convenção, vamos apelidar o streamlit de st
import pandas as pd # Importando a biblioteca Pandas
import altair as alt # Importando a biblioteca

df = pd.read_csv('NSE-TATAGLOBAL11.csv', sep=',') # Importando a base de dados e transformando em um DataFrame do pandas

# Aqui definimos o título da página e o layout como wide
st.set_page_config(page_title="Meu Dashboard",layout="wide")

#Título do seu dashboard
st.write("""
# Meu primeiro Dashboard
Abaixo veremos os próximos passos
""")


# Importando as demais bibliotecas necessárias
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Manipulando o dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# Setando o index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
new_data.sort_index(ascending=True, inplace=True)

# Criando a base de treino e test
dataset = new_data.values
train = dataset[0:987,:]
valid = dataset[987:,:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# Setando o modelo
model = Sequential()
model.add(LSTM(units=50,  return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

# Dados de treino e predição
train = new_data[:987]
valid = new_data[987:]
train['date'] = train.index
valid['date'] = valid.index
valid['Predictions'] = closing_price



#Título acima do gráfico
st.write("""
 Previsão de Ações
""") 

# Definição do gráfico
stocks_train = alt.Chart(train).mark_line().encode(
    x='date',
    y='Close'
)
stocks_valid = alt.Chart(valid).mark_line(color="green").encode(
    x='date',
    y='Close'
)
stocks_pred = alt.Chart(valid).mark_line(color="red").encode(
    x='date',
    y='Predictions'
)

# Plotagem do gráfico
st.altair_chart(stocks_train.interactive() + stocks_valid.interactive() + stocks_pred.interactive(), use_container_width=True)