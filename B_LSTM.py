import numpy as np
import random as rn
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Input, Model
from keras import optimizers
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras import backend as K
from pandas import DataFrame 
from pandas import concat
import time



###################### TRAINING #############################
#----------------------------------------------------------
#---- multivariate LSTM forecasting  ----
#----------------------------------------------------------
# predicting streamflow discharge for furture n_out time steps given meteorogical data at previous n_in time step.
# --- convert series to supervised learning
def series_to_supervised(data_in, data_out, n_in=1, n_out=1, dropnan=True):
	n_vars_in = 1 if type(data_in) is list else data_in.shape[1]
	n_vars_out = 1 if type(data_out) is list else data_out.shape[1]
	df_in = DataFrame(data_in)
	df_out = DataFrame(data_out)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df_in.shift(i))
		names += [('var_in%d(t-%d)' % (j+1, i)) for j in range(n_vars_in)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df_out.shift(-i))
		if i == 0:
			names += [('var_out%d(t)' % (j+1)) for j in range(n_vars_out)]
		else:
			names += [('var_out%d(t+%d)' % (j+1, i)) for j in range(n_vars_out)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

 
#----------------------------------------------------------
#---- Load data ----
#----------------------------------------------------------
ndays = 60  
nfuture = 1 
ninputs = 3
nobs = ndays * ninputs
Ntest = 365*19+5

tmp = np.loadtxt('Data_98_18.dat')

xtmp = tmp[:,0:ninputs]
ytmp = tmp[:,-nfuture]
ytmp = np.reshape(ytmp, (-1, 1))
print('max of obs (in/d)', np.max(ytmp))


# ======= Train on multiple lag timestep ============
# 1.1 --- formulate supervised learning problem
reframed = series_to_supervised(xtmp, ytmp, ndays, nfuture)
print('Shape of supervised dataset: ', np.shape(reframed))

XYdata = reframed.values

# 1.2 ---split into train and test sets
XYtrain = XYdata[:-Ntest, :]
XYtest = XYdata[-Ntest:, :]

yobs_train = XYdata[:-Ntest, -nfuture]
yobs_test = XYdata[-Ntest:, -nfuture]
print('shape of yobs_train and yobs_test is ', yobs_train.shape, yobs_test.shape)
print('min and max of yobs_test', np.min(yobs_test), np.max(yobs_test))

# 1.3 ---scale training and testing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaledXYtrain = scaler.fit_transform(XYtrain)
scaledXYtest = scaler.transform(XYtest)
print('shape of scaledXYtrain and scaledXYtest is ', scaledXYtrain.shape, scaledXYtest.shape)

# 1.4 ---split into input and outputs
train_X, train_y = scaledXYtrain[:, :nobs], scaledXYtrain[:, -nfuture]
test_X = scaledXYtest[:, :nobs]
print('shape of train_X, train_y, and test_X: ', train_X.shape, train_y.shape, test_X.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], ndays, ninputs))
test_X = test_X.reshape((test_X.shape[0], ndays, ninputs))
print('shape of train_X and test_X in 3D: ', train_X.shape, test_X.shape)


#----------------------------------------------------------
#---- Train LSTM model  ----
#----------------------------------------------------------
Nepoch = 100
dropout = 0.1
Nmc = 100

rmse_train = np.zeros((Nepoch,1))
r2_train = np.zeros((Nepoch,1))
ave_std_train = np.zeros((Nepoch,1))
rmse_test = np.zeros((Nepoch,1))
r2_test = np.zeros((Nepoch,1))
ave_std_test = np.zeros((Nepoch,1))

input_shape=(train_X.shape[1], train_X.shape[2])

# 1.3 ---define and fit LSTM model
inp = Input(input_shape)
x = LSTM(20, recurrent_dropout = dropout)(inp,training=True)
out = Dense(nfuture)(x)
model = Model(inputs=inp, output=out)

adam = optimizers.adam(lr=0.001)  
model.compile(loss='mse', optimizer=adam)

start = time.time()
for i in range(Nepoch):
	model.fit(train_X, train_y, epochs=1, batch_size=10, verbose=0)

	print('---------------- Epoch %d -------------' %i)
	mc_train = []
	mc_test = []
	# MC loop
	for imc in range(Nmc):
	#-- evaluate training data
		yhat = model.predict(train_X)
		train_X = train_X.reshape((train_X.shape[0], nobs))
		inv_yhat = np.concatenate((train_X,yhat), axis=1)
		inv_yhat = scaler.inverse_transform(inv_yhat)
		NNytrain = inv_yhat[:,-nfuture]
		mc_train.append(NNytrain)
		train_X = train_X.reshape((train_X.shape[0], ndays, ninputs))

	#-- evaluate testing data
		yhat_test = model.predict(test_X)
		test_X = test_X.reshape((test_X.shape[0], nobs))
		tmp_test = np.concatenate((test_X, yhat_test), axis=1)
		inv_yhat_test = scaler.inverse_transform(tmp_test)
		NNytest = inv_yhat_test[:,-nfuture]
		mc_test.append(NNytest)
		test_X = test_X.reshape((test_X.shape[0], ndays, ninputs))

	mc_train = np.array(mc_train)
	print('shape of mc_train is', mc_train.shape)
	std_train = mc_train.std(axis=0)
	ave_std_train[i] = std_train.mean()
	mean_train = mc_train.mean(axis=0)
	rmse_train[i] = sqrt(mean_squared_error(yobs_train, mean_train))
	r2_train[i] = r2_score(yobs_train, mean_train)	
	print('Training -------: rmse and r2 is %6.3f and %6.4f' % (rmse_train[i],r2_train[i]))

	mc_test = np.array(mc_test)
	print('shape of mc_test is', mc_test.shape)
	std_test = mc_test.std(axis=0)
	ave_std_test[i] = std_test.mean()
	mean_test = mc_test.mean(axis=0)
	rmse_test[i] = sqrt(mean_squared_error(yobs_test, mean_test))
	r2_test[i] = r2_score(yobs_test, mean_test)
	print('Testing--------------------: rmse and r2 is %6.3f and %6.4f' % (rmse_test[i],r2_test[i]))

end = time.time()
print('Epoch takes time: %f', end-start)


# #----- Save results
np.savetxt('R2_Train2_drop01.out',np.concatenate((rmse_train, r2_train, ave_std_train), axis=1))
np.savetxt('R2_Test19_drop01.out',np.concatenate((rmse_test, r2_test, ave_std_test), axis=1))

np.savetxt('Sample_Train2_drop01.out',mc_train)
np.savetxt('Sample_Test19_drop01.out',mc_test)


