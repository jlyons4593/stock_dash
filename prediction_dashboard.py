import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Prediction Dashboard')

input = st.text_input('Input stock ticker', 'GOLD')
df = data.DataReader(input,'yahoo',start , end)

st.subheader('Daily Close Price vs Time')
fig = plt.figure(figsize=(20,5))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Daily Close Price with 100 day moving average vs Time')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(20,5))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Daily Close Price with 100, 200 day moving average vs Time')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(20,5))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

train_ind = int(len(df)*0.8)
train = pd.DataFrame(df['Close'][:train_ind])
test = pd.DataFrame(df['Close'][train_ind:])


scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)

x_train = []
y_train = []

for i in range(100,train_scaled.shape[0]):
    x_train.append(train_scaled[i-100:i])
    y_train.append(train_scaled[i,0])

x_train , y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')

last_100_Days=train.tail(100)
final_df = last_100_Days.append(test, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted_model= model.predict(x_test)

scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_predicted = y_predicted_model * scaler_factor
y_test = y_test * scaler_factor


st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(20,5))
plt.plot(y_test, 'b', label='original price')
plt.plot(y_predicted, 'r', label='predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
