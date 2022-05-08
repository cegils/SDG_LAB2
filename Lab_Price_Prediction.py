### Imports
## Basic
import pandas as pd
import numpy as np
#%matplotlib ipympl
import matplotlib.pyplot as plt
import datetime
import datetime as dt
## Neural Networks and Machine Learning
#%tensorflow_version 1.x #This is a Google Collab Magic Word not to be used in Spyder
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, SGD
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
#from sklearn.linear_model import LinearRegression, Ridge
from tensorflow import keras

###End Imports

###Functions

## Upload Solar, wind and Belpex Data
def upload_files(): 

    ##Global Variables
    global df_solar
    global df_wind
    global df_belpex    

    ## Solar DataFrame
    df_solar = pd.read_csv(r'solar_20162017.csv', header=0) #Upload File
    df_solar = df_solar.rename(columns = {'Unnamed: 0':'time'})#Rename the first column
    df_solar['time'] = pd.to_datetime(df_solar['time'])#Change format of time to datetime
    df_solar.set_index('time', inplace=True) #Set time column as index of the DataFrame
    df_solar = df_solar.LoadFactor # Set the dataframe df_solar to only the LoadFactor column + Index

    #print(df_solar)

    ## Wind DataFrame
    df_wind = pd.read_csv(r'wind_20162017.csv', header=0) #Upload File
    df_wind = df_wind.rename(columns = {'Unnamed: 0':'time'})#Rename the first column
    df_wind['time'] = pd.to_datetime(df_wind['time'])#Change format of time to datetime
    df_wind.set_index('time', inplace=True) #Set time column as index of the DataFrame
    df_wind = df_wind.LoadFactor # Set the dataframe df_wind to only the LoadFactor column + Index

    #print(df_wind)

    ## Wind DataFrame
    df_wind = pd.read_csv(r'wind_20162017.csv', header=0) #Upload File
    df_wind = df_wind.rename(columns = {'Unnamed: 0':'time'})#Rename the first column
    df_wind['time'] = pd.to_datetime(df_wind['time'])#Change format of time to datetime
    df_wind.set_index('time', inplace=True) #Set time column as index of the DataFrame
    df_wind = df_wind.LoadFactor # Set the dataframe df_wind to only the LoadFactor column + Index

    #print(df_wind)

    ## Belpex DataFrame
    df_belpex = pd.read_csv(r'belpex_20162017.csv', header=0) #Upload File
    df_belpex = df_belpex.rename(columns = {'Unnamed: 0':'time'})#Rename the first column
    df_belpex['time'] = pd.to_datetime(df_belpex['time'])#Change format of time to datetime
    df_belpex.set_index('time', inplace=True) #Set time column as index of the DataFrame

    #print(df_belpex)
    
    return "Files Uploaded"

##Get Accuracy (Not used)
def get_accuracy(x, y): # Get accuracy of data with respect to the predictions
    return np.mean(np.abs(x - y))/np.mean(x)

##Create and Compile model with predefined Neurons
def create_model(neurons):
    
    #Model Creation
    activation_functions = ['relu', 'linear']#['relu', 'linear']
    # Input 1: Solar Load Factor
    input_1 = keras.Input(shape=(7,24),name="solar_input")
    # Input 2: Wind Load Factor
    input_2 = keras.Input(shape=(7,24),name="wind_input")
    # Input 3: Prices Last Week
    input_3 = keras.Input(shape=(7,24),name="plweek_input")
    #List of Inputs
    inputs = layers.Concatenate(axis=-1)([input_1, input_2, input_3])
    # Create the connection between the layers
    layer_1 = layers.Dense(neurons[0], activation=activation_functions[0], name="input_layer")(inputs)
    layer_2 = layers.Dense(neurons[1], activation=activation_functions[0], name="layer_2")(layer_1)
    outputs = layers.Dense(neurons[2], activation=activation_functions[1], name="output_layer")(layer_2)
    # Creation of the model
    model = keras.Model(inputs=[input_1, input_2, input_3], outputs=outputs, name="neural_network_1")
    print(model.summary()) # Print model summary
    # Model Compilation     
    rprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    model.compile(loss='mean_squared_error', optimizer=rprop, metrics=['accuracy'])
    return model

##Train model with predefined parameters and datasets
def train_model(model,s_train,w_train,p_train,p_target_train,batch_size, epochs, validation_split):
    print("Start Training Model") # Print model summary
    output_training = model.fit(
        {"solar_input": s_train, "w_input": w_train, "plweek_input": p_train}, #Inputs
        {"output_layer": p_target_train}, #Outputs
        batch_size=n_batch,
        epochs=n_epochs,
        validation_split=validation_split,
        verbose=0
    )
    print("Model Trained") # Print model summary
    return output_training

## End functions



### Exercise 2 Code

## Start and End date for DataFrames of Solar, Wind and Belpex (prices)

start_date = dt.datetime(2016,1,1)
end_date = dt.datetime(2017,12,31)

## Upload Solar, Wind and Belpex
upload_files() 
#print("DataFrame Solar:",df_solar)
#print("DataFrame Wind:",df_wind)
#print("DataFrame Belpex:",df_belpex)



## Fixing and Reframing Belpex data to quarter hour (15min)

dates = pd.date_range(start=start_date, end=end_date, freq='1H')

for d in dates:
    try:
        p = df_belpex.loc[d] #Find any non-null value
    except KeyError:
        df_belpex.loc[d] = df_belpex.loc[d-dt.timedelta(hours=1)] # In case of finding a null value, replace with the previous data
df_belpex = df_belpex.sort_index() # Reorganize the dataframe
df_belpex = df_belpex[start_date:end_date] #Reasign the start and end date
df_wind =  df_wind.fillna(method='pad') # Fill in the NaN spaces
df_belpex = df_belpex.resample('15T').pad() # Resample the data to 15min 

d = {'belpex': df_belpex.values.flatten(), 'solar': df_solar.values, 'wind': df_wind.values} #Transform data into a dictionary with 3 elements
data = pd.DataFrame(index=df_belpex.index, data=d) #Make an independent dataframe using data "d" and the index of belpex (time)
data_dummy = pd.DataFrame(index=df_belpex.index, data=d) #dummy for comparison

##Showing figures for obtained data
start = datetime.datetime(2016, 1, 1, 0, 0) #New start and end date
end = datetime.datetime(2016, 1, 14, 23, 45)

#Data from solar, wind and belpex
plt.figure()
plt.subplot(311)
plt.plot(data.belpex[start:end], label='belpex')
plt.legend(frameon=False)
plt.subplot(312)
plt.plot(data.solar[start:end], label='solar')
plt.legend(frameon=False)
plt.subplot(313)
plt.plot(data.wind[start:end], label='wind')
plt.legend(frameon=False)
plt.show()

#Data from belpex 2016-2017
plt.figure()
data.belpex.plot(grid=True)
plt.show()

##Removing outliers of belpex (to increase accuracy of the training set)
mean = data.belpex.mean() #  Mean or average of the sample
std = data.belpex.std() # Standard deviation
n_std = 5 # Multiplier
data['belpex'][(data.belpex >= mean + n_std*std)] = mean + n_std*std # values greater or equal than the mean + n_std*std will have a saturation
data['belpex'][(data.belpex <= mean - n_std*std)] = mean + n_std*std # values less or equal than the mean + n_std*std will have a saturation

#Show comparison
plt.figure()
plt.subplot(211)
plt.plot(data.belpex, label='belpex now')
plt.legend(frameon=False)
plt.subplot(212)
plt.plot(data_dummy.belpex, label='belpex old')
plt.legend(frameon=False)
plt.show()

# Autocorrelation between measurements of belpex
n_days = 30
lags = np.arange(1, 96*n_days)
acors = []
for lag in lags:
    acors.append(data.belpex.autocorr(lag))
plt.figure()
plt.plot(lags/4/24.0, acors)
plt.xlabel('Time lag in days')
plt.grid(True)

#Scatter Matrix
s_matrix = pd.plotting.scatter_matrix(data[['belpex', 'solar', 'wind']])

#Linear Regression Model
#lr_model = LinearRegression(normalize=True) # Approximation of the problem using a linera regression

##Training Data

lags = list(range(96, 96*365, 96*7)) # Create the lags of the time series
features = ['wind', 'solar']
#print(lags)
#len(lags)

index = data.index 
n_hours = 24
start_date_target = datetime.datetime(2017, 1, 2, 0, 0)
end_date_target = datetime.datetime(2017, 1, 8, 23, 45)
start_date_input_res = datetime.datetime(2017, 1, 2, 0, 0) #Start/End Dates for Prices (one day shift)
end_date_input_res = datetime.datetime(2017, 1, 8, 23, 45)
start_date_input_prices = datetime.datetime(2017, 1, 1, 0, 0) #Start/End Dates for Prices (one day shift)
end_date_input_prices = datetime.datetime(2017, 1, 7, 23, 45)
train_prices=[]
train_solar=[]
train_wind=[]
train_target_price=[]


for lag in lags:
    data['belpex_lag_{}'.format(lag)] = data.belpex.shift(lag)
    data['solar_lag_{}'.format(lag)] = data.solar.shift(lag)
    data['wind_lag_{}'.format(lag)] = data.wind.shift(lag)
    data['target_lag_{}'.format(lag)] = data.belpex.shift(lag)
    train_prices.append(data['belpex_lag_{}'.format(lag)][start_date_input_prices:end_date_input_prices].resample('1H').mean().values.reshape(-1, n_hours))
    train_solar.append(data['solar_lag_{}'.format(lag)][start_date_input_res:end_date_input_res].resample('1H').mean().values.reshape(-1, n_hours))
    train_wind.append(data['wind_lag_{}'.format(lag)][start_date_input_res:end_date_input_res].resample('1H').mean().values.reshape(-1, n_hours))
    train_target_price.append(data['target_lag_{}'.format(lag)][start_date_target:end_date_target].resample('1H').mean().values.reshape(-1, n_hours))

#Convert lists to arrays
train_prices = np.array(train_prices)
train_solar = np.array(train_solar)
train_wind = np.array(train_wind)
train_target_price = np.array(train_target_price)


data['day_in_year'] = data.index.dayofyear ##Not used
data = data.loc[index, :]
data = data.dropna()


# Split data in Training and test Data (Not that necessary, using validation split directly in .fit method)

s_train, s_test, w_train, w_test = train_test_split(train_solar, train_wind, test_size=0.10, random_state=42) # Split into solar and wind train/validation sets
p_train, p_test, p_target_train, p_target_test = train_test_split(train_prices, train_target_price, test_size=0.10, random_state=42) # Split into prices prevoious dates train/validation sets

## Neural Network 1

neurons = [50, 50, 24] # [Input Layer, Hidden Layer 2, Outputs]
n_epochs = 1200 # Number of interactions
n_batch = 10 # Number of batches
val_size = 0.3

#Model Creation

model_1 = create_model(neurons)   

#Model Training

output_training_1 = train_model(model_1, s_train, w_train, p_train, p_target_train, n_batch, n_epochs, val_size)

mse_1 = output_training_1.history['loss'][-1]
print('- Mean squared error for ' + model_1.name + ' is %.4f' % mse_1 + ' @ Iteration ' + str(len(output_training_1.history['loss'])))

## Neural Network 2

neurons_2 = [60, 48, 24] # [Input Layer, Hidden Layer 2, Outputs]
n_epochs_2 = 1200 # Number of interactions
n_batch_2 = 10 # Number of batches

#Model Creation

model_2 = create_model(neurons_2)   

#Model Training

output_training_2 = train_model(model_2, s_train, w_train, p_train, p_target_train, n_batch_2, n_epochs_2, val_size)

## Neural Network 3

neurons_3 = [72, 150, 24] # [Input Layer, Hidden Layer 2, Outputs]
n_epochs_3 = 1200 # Number of interactions
n_batch_3 = 10 # Number of batches

#Model Creation

model_3 = create_model(neurons_3)   

#Model Training

output_training_3 = train_model(model_3, s_train, w_train, p_train, p_target_train, n_batch_2, n_epochs_3, val_size)

mse_1 = output_training_1.history['loss'][-1]
mse_2 = output_training_2.history['loss'][-1]
mse_3 = output_training_3.history['loss'][-1]


print('- Mean squared error for Neural Network 1 is %.4f' % mse_1 + ' @ Iteration ' + str(len(output_training_1.history['loss'])))
print('- Mean squared error for Neural Network 2 is %.4f' % mse_2 + ' @ Iteration ' + str(len(output_training_2.history['loss'])))
print('- Mean squared error for Neural Network 3 is %.4f' % mse_3 + ' @ Iteration ' + str(len(output_training_3.history['loss'])))

# Plots Loss
epc = range(1, n_epochs + 1)
epc_2 = range(1, n_epochs_2 + 1)
epc_3 = range(1, n_epochs_3 + 1)

plt.figure()
plt.plot(epc,output_training_1.history['loss'], color='red', label='Loss Model 1')
plt.plot(epc_2, output_training_2.history['loss'], color='blue', label='Loss Model 2')
plt.plot(epc_3, output_training_3.history['loss'], color='black', label='Loss Model 3')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(frameon=True)
plt.show()

## Price Prediction Model 1 vs Model 2 vs Model 3

#RES Data Frames for testing
solar_pred = s_test
wind_pred = w_test
#PRICES Data Frames for Testing
prices_pred = p_test
actual_price = p_target_test

predict_nn_model_1 = model_1.predict({"solar_input": solar_pred, "w_input": wind_pred, "plweek_input": prices_pred})
predict_nn_model_2 = model_2.predict({"solar_input": solar_pred, "w_input": wind_pred, "plweek_input": prices_pred})
predict_nn_model_3 = model_3.predict({"solar_input": solar_pred, "w_input": wind_pred, "plweek_input": prices_pred})



## Plots
plt.figure()
plt.subplot(311)
plt.plot(p_target_test[:1,:7,:].flatten(), color='blue', label='actual price')
plt.plot(predict_nn_model_1[:1,:7,:].flatten(), color='red', label='forecast Model 1')
plt.xlabel('Hour of the Day')
plt.ylabel('Prices')
plt.legend(frameon=True)
plt.subplot(312)
plt.plot(p_target_test[:1,:7,:].flatten(), color='blue', label='actual price')
plt.plot(predict_nn_model_2[:1,:7,:].flatten(), color='green', label='forecast Model 2')
plt.xlabel('Hour of the Day')
plt.ylabel('Prices')
plt.legend(frameon=True)
plt.subplot(313)
plt.plot(p_target_test[:1,:7,:].flatten(), color='blue', label='actual price')
plt.plot(predict_nn_model_3[:1,:7,:].flatten(), color='black', label='forecast Model 3')
plt.xlabel('Hour of the Day')
plt.ylabel('Prices')
plt.legend(frameon=True)
plt.show()

print('- Mean squared error for Neural Network 1 is %.4f' % mse_1 + ' @ Iteration ' + str(len(output_training_1.history['loss'])))
print('- Mean squared error for Neural Network 2 is %.4f' % mse_2 + ' @ Iteration ' + str(len(output_training_2.history['loss'])))
print('- Mean squared error for Neural Network 3 is %.4f' % mse_3 + ' @ Iteration ' + str(len(output_training_3.history['loss'])))

##Print Results to csv

pred_1 = predict_nn_model_3[:1,:1,:].reshape(24).tolist()
pred_2 = predict_nn_model_3[:1,1:2,:].reshape(24).tolist()
pred_3 = predict_nn_model_3[:1,2:3,:].reshape(24).tolist()
dat = [*pred_1, *pred_2, *pred_3]
dat_1= pd.DataFrame()
dat_1['Prices'] = pd.DataFrame(dat)
header = ["Prices"]
dat_1.to_csv('predictions.csv', columns = header,index=False)

#End Mana

### End Exercise 2















