import pandas as pd
from pandas import DataFrame 
from pandas import concat
import numpy as np
import random as rn
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras import optimizers
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.models import model_from_json
from keras import backend as K
import time



###################### TRAINING #############################
#----------------------------------------------------------
#---- multivariate LSTM forecasting  ----
#----------------------------------------------------------
# predicting NEE for furture n_out time steps given meteorogical data at previous n_in time step.
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
ninputs = 4
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

# 1.3 ---scale training data both X and y
scaler = MinMaxScaler(feature_range=(0, 1))
scaledXYtrain = scaler.fit_transform(XYtrain)
scaledXYtest = scaler.transform(XYtest)
print('shape of scaledXYtrain and scaledXYtest is ', scaledXYtrain.shape, scaledXYtest.shape)

# 1.4 ---split into input and outputs
train_X, train_y = scaledXYtrain[:, :nobs], scaledXYtrain[:, -nfuture]
test_X = scaledXYtest[:, :nobs]
print('shape of train_X, train_y, and test_X: ', train_X.shape, train_y.shape, test_X.shape)
# print('min and max of train_X', np.min(train_X,axis=0), np.max(train_X,axis=0))
# print('min and max of train_y', np.min(train_y,axis=0), np.max(train_y,axis=0))
# print('min and max of test_X', np.min(test_X,axis=0), np.max(test_X,axis=0))


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], ndays, ninputs))
test_X = test_X.reshape((test_X.shape[0], ndays, ninputs))
print('shape of train_X and test_X in 3D: ', train_X.shape, test_X.shape)

# exit()

# 1.3 ---define and fit LSTM model
Nepoch = 100
Nnode = 20

rmse_train = np.zeros(Nepoch)
r2_train = np.zeros(Nepoch)
rmse_test = np.zeros(Nepoch)
r2_test = np.zeros(Nepoch)
NNytrain =np.zeros((train_X.shape[0],Nepoch))
NNytest =np.zeros((test_X.shape[0],Nepoch))

# 1.3 ---define and fit LSTM model
model = Sequential()
model.add(LSTM(Nnode, input_shape=(train_X.shape[1], train_X.shape[2])))#, return_sequences=True))
# model.add(LSTM(10))
model.add(Dense(nfuture)) 
adam = optimizers.adam(lr=0.001)  #default lr=0.001
model.compile(loss='mse', optimizer=adam)

for i in range(Nepoch):
	model.fit(train_X, train_y, epochs=1, batch_size=10, verbose=0, shuffle=True)
	yhat_train = model.predict(train_X)
	train_X = train_X.reshape((train_X.shape[0], nobs))
	tmp1 = np.concatenate((train_X,yhat_train), axis=1)
	inv_yhat_train = scaler.inverse_transform(tmp1)
	NNytrain[:,i] = inv_yhat_train[:,-nfuture]
	rmse_train[i] = sqrt(mean_squared_error(yobs_train, NNytrain[:,i]))
	r2_train[i] = r2_score(yobs_train, NNytrain[:,i])
	
	train_X = train_X.reshape((train_X.shape[0], ndays, ninputs))
	
	# print('Training data: rmse and r2 of Epoch%d is %6.3f and %6.4f' % (i,rmse_train[i],r2_train[i]))

	#----- testing data ----
	yhat_test = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], nobs))
	tmp2 = np.concatenate((test_X,yhat_test), axis=1)
	inv_yhat_test = scaler.inverse_transform(tmp2)
	NNytest[:,i] = inv_yhat_test[:,-nfuture]
	rmse_test[i] = sqrt(mean_squared_error(yobs_test, NNytest[:,i]))
	r2_test[i] = r2_score(yobs_test, NNytest[:,i])

	test_X = test_X.reshape((test_X.shape[0], ndays, ninputs))
	
	# print('------- Testing data: rmse and r2 of Epoch%d is %6.3f and %6.4f' % (i,rmse_test[i],r2_test[i]))

min_rmse_train = np.amin(rmse_train)
min_rmse_train_inx = np.argmin(rmse_train)
max_r2_train = np.amax(r2_train)
max_r2_train_inx = np.argmax(r2_train)

min_rmse_test = np.amin(rmse_test)
min_rmse_test_inx = np.argmin(rmse_test)
max_r2_test = np.amax(r2_test)
max_r2_test_inx = np.argmax(r2_test)

print('---------------------- ndays %d -----------------------' % ndays)
print('rmse of training data is %6.3f with index %3d' % (min_rmse_train,min_rmse_train_inx))
print('R2 of training data is %6.4f with index %3d' % (max_r2_train,max_r2_train_inx))
print('-------------------------------------------------------')
print('rmse of testing data is %6.3f with index %3d' % (min_rmse_test,min_rmse_test_inx))
print('R2 of testing data is %6.4f with index %3d' % (max_r2_test,max_r2_test_inx))
print('-------------------------------------------------------')
print('For best fitting, rmse of testing data is %6.3f' % rmse_test[max_r2_train_inx])
print('For best fitting, R2 of testing data is %6.4f' % r2_test[max_r2_train_inx])
print('-------------------------------------------------------')
print('At end of Epoch, rmse of traing and testing data is %6.2f and %6.2f' % (rmse_train[-1],rmse_test[-1]))
print('At end of Epoch, R2 of traing and testing data is %6.2f and %6.2f' % (r2_train[-1],r2_test[-1]))


np.savetxt('R2_Train2_N%d_LB%d.out' %(Nnode,ndays), [rmse_train,r2_train])
np.savetxt('R2_Test19_N%d_LB%d.out' %(Nnode,ndays), [rmse_test,r2_test])

np.savetxt('Y_Train2_N%d_LB%d.out' %(Nnode,ndays), NNytrain)
np.savetxt('Y_Test19_N%d_LB%d.out' %(Nnode,ndays), NNytest)

plt.plot(r2_train,'b')
plt.plot(r2_test,'r')
plt.savefig('R2_Train2Test19_N%d_LB%d.png' %(Nnode,ndays), bbox_inches='tight',pad_inches=0, dpi=300)
plt.show()







