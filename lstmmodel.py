#importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# make sure dataset imported has date in ascending order (Start from oldest date few years back)
df = pd.read_csv("tataglobal.csv")

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

print(new_data)
print(new_data.shape)

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#dataset consists of closing prices according to day
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]
print(len(train))

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset) #scales data accordingly to into value ranges from 0 to 1
print("Scaled Data:")
print(scaled_data)

#Goes back 60 days to cross-check with data
#x_train features contains price of past 60 days, model will draw links between these values
# In other words, there are 60 "features" for the model
#y_train is the price on the 60th day

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
print("x_train:")
print(y_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1))) #x_train.shape[1] = 60, meaning that the model looks at 60 days of data
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using values from validation data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
print("input shape")
print(inputs.shape)

X_valid = []

#start from 60th value in inputs
for i in range(60,inputs.shape[0]):
    X_valid.append(inputs[i-60:i,0])
X_valid = np.array(X_valid)
print(X_valid.shape)

X_valid = np.reshape(X_valid, (X_valid.shape[0],X_valid.shape[1],1))
print(X_valid.shape)
closing_price = model.predict(X_valid)
closing_price = scaler.inverse_transform(closing_price)

print(closing_price.shape)
print("Valid Shape")
print(valid.shape)
#print(closing_price)
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print(rms)

# plot graph
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price

plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(valid['Predictions'])
