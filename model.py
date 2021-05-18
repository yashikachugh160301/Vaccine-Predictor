#importing libraries
import numpy as np
import pandas as pd
import statsmodels as sm
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import datetime
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
#loading data
url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/India.csv"
df = pd.read_csv(url,index_col=0,parse_dates=[0])
#df.info()
df=df.reset_index()
df=df.drop(columns=['vaccine','source_url','total_vaccinations','location'])
df['date']=pd.to_datetime(df['date'])

total= df['people_vaccinated'][len(df)-1]

df=df.set_index('date').diff()

df=df.iloc[1:,:]

# preparing independent and dependent features
def prepare_data(timeseries_data, n_features):
	X, y =[],[]
	for i in range(len(timeseries_data)):
		# find the end of this pattern
		end_ix = i + n_features
		# check if we are beyond the sequence
		if end_ix > len(timeseries_data)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

df=df.reset_index()

# define input sequence
timeseries_data = df.iloc[:,1]
# choose a number of time steps
n_steps = 7
# split into samples
X, y = prepare_data(timeseries_data, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X_re = X.reshape((X.shape[0], X.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(loss='mae',optimizer='Adam')
# fit model
model.fit(X_re, y, epochs=300, verbose=1)

# demonstrate prediction for next few days
x_input = X[-1]
temp_input=list(x_input)
lst_output=[]
i=0
yhat=0
while(i<10): #next 30 days
    
    if(len(temp_input)>n_steps):
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        #print(x_input)
        x_input = x_input.reshape((1, n_steps, n_features))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.append(yhat[0][0])
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.append(yhat[0][0])
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        
        #print(yhat[0])
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        
        i=i+1
        
day_new=np.arange(1,len(timeseries_data)+1)
day_pred=np.arange(len(timeseries_data),len(timeseries_data)+10)

plt.plot(day_new,timeseries_data)
plt.plot(day_pred,lst_output)
plt.title('Covid Vaccination-India')
plt.xlabel('Number of days starting from 16 feb')
plt.ylabel('People Vaccinated (in 100 millions)')
plt.savefig('pred.png')

pred=np.sum(lst_output)

from matplotlib.figure import Figure
fig = Figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(day_new,timeseries_data)
axis.plot(day_pred,lst_output)
        
axis.set_title('Covid Vaccination-India')
axis.set_xlabel('Number of days starting from 16 Jan')
axis.set_ylabel('People Vaccinated (in 100 millions)')
axis.show()
    
create_plot()
