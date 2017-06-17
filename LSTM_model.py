import os
import settings
import pandas as pd
import numpy as np
# fix random seed for reproducibility
np.random.seed(settings.SEED)
from math import sqrt
import matplotlib
# be able to save images on server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	# combine input and output
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	#return output
	return inverted[0, -1]

def fit_lstm(train, batch_size, epochs, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(epochs):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# Update LSTM model
def update_model(model, train, batch_size, epochs):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	for i in range(epochs):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()


def evaluate_model(scaler, train_scaled, test_scaled, raw_values, batch_size=1, epochs=2000, neurons=2, updates=0, seed=True):
	# fit the model
	lstm_model = fit_lstm(train_scaled, batch_size, epochs, neurons)
	# forecast the entire training dataset to build up state for forecasting
	if seed:
		train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
		lstm_model.predict(train_reshaped, batch_size=batch_size)
	# walk-forward validation on validation set
	train_copy = np.copy(train_scaled)
	predictions = list()
	for i in range(len(test_scaled)):
		# update model
		if i > 0 and updates>0:
			update_model(lstm_model, train_copy, batch_size, updates)
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
		# add to training set
		train_copy = np.concatenate((train_copy, test_scaled[i,:].reshape(1, -1)))
		# print outputs
		expected = raw_values[len(train_scaled) + i + 1]
		#print('>Predicted=%f, Expected=%f' % (yhat, expected))

	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-len(test_scaled):], predictions))
	#print('RMSE: %.3f' % rmse)

	return rmse, predictions


'''
# load data
series = pd.Series.from_csv(os.path.join(settings.PROCESSED_DIR, 'train.csv'))

# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
# split data into train and validation-sets
train, validation = supervised_values[0:-12], supervised_values[-12:]
# transform the scale of the data
scaler, train_scaled, validation_scaled = scale(train, validation)

# fit the model
#evaluate_model(scaler, train_scaled, validation_scaled, raw_values)

# experiment
# vary training update epochs
epochs = [0, 2, 5, 10, 20]
for e in epochs:
	rmse, predictions = evaluate_model(scaler, train_scaled, validation_scaled, raw_values, updates=e)
	print('>Updates=%d, RMSE=%.3f' % (e, rmse))


# stateless vs Stateful 
rmse, predictions = evaluate_model(scaler, train_scaled, validation_scaled, raw_values, seed=True, updates=0)
print('>with-seed, RMSE=%.3f' % (rmse))
rmse, predictions = evaluate_model(scaler, train_scaled, validation_scaled, raw_values, seed=False, updates=0)
print('>without-seed, RMSE=%.3f' % (rmse))


# vary training epochs
epochs = [500, 1000, 2000, 3000]
for e in epochs:
	rmse, predictions = evaluate_model(scaler, train_scaled, validation_scaled, raw_values, epochs=e)
	print('>Epochs=%d, RMSE=%.3f' % (e, rmse))


# vary training epochs
batches = [1, 2, 4]
for e in batches:
	rmse, predictions = evaluate_model(scaler, train_scaled, validation_scaled, raw_values, batch_size=e)
	print('>Batches=%d, RMSE=%.3f' % (e, rmse))



# vary training neurons
neurons = [1, 2, 3, 4, 5]
for e in neurons:
	rmse, predictions = evaluate_model(scaler, train_scaled, validation_scaled, raw_values, neurons=e)
	print('>Neurons=%d, RMSE=%.3f' % (e, rmse))
'''

'''
# Build final model
# load data
train = pd.Series.from_csv(os.path.join(settings.PROCESSED_DIR, 'train.csv'))

# transform data to be stationary
raw_values = train.values
diff_values = difference(raw_values, 1)
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
# fit scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(supervised_values)
# transform train
reshaped_values = supervised_values.reshape(supervised_values.shape[0], supervised_values.shape[1])
train_scaled = scaler.transform(reshaped_values)

# Save final model (combine train and validation)
lstm_model = fit_lstm(train_scaled, batch_size=1, epochs=2000, neurons=2)

# save model to file
# serialize model to JSON
model_json = lstm_model.to_json()
with open(os.path.join(settings.OUTPUT_DIR, "lstm_model.json"), "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
lstm_model.save_weights(os.path.join(settings.OUTPUT_DIR, "lstm_model.h5"))
'''



# Validate model
# load json and create model
json_file = open(os.path.join(settings.OUTPUT_DIR, "lstm_model.json"), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(settings.OUTPUT_DIR, "lstm_model.h5"))
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam')


# load and prepare datasets
train = pd.Series.from_csv(settings.PROCESSED_DIR + 'train.csv')
test = pd.Series.from_csv(settings.PROCESSED_DIR + 'test.csv')
series = pd.concat([train, test])
# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
# split data into train and validation-sets
supervised_train, supervised_test = supervised_values[0:-len(test)], supervised_values[-len(test):]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(supervised_train, supervised_test)


batch_size=1
#updates=2

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
loaded_model.predict(train_reshaped, batch_size=batch_size)
# walk-forward validation on validation set
#train_copy = np.copy(train_scaled)
predictions = list()
for i in range(len(test_scaled)):
	# update model
	#if i > 0:
		#update_model(loaded_model, train_copy, batch_size, updates)
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(loaded_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	# add to training set
	#train_copy = np.concatenate((train_copy, test_scaled[i,:].reshape(1, -1)))
	# print outputs
	expected = raw_values[len(train_scaled) + i + 1]
	print('>Predicted=%f, Expected=%f' % (yhat, expected))
# report performance
rmse = sqrt(mean_squared_error(raw_values[-len(test_scaled):], predictions))
print('RMSE: %.3f' % rmse)

plt.plot(test.values)
plt.plot(predictions, color='red')
plt.title('LSTM Model for test set')
plt.savefig(os.path.join(settings.OUTPUT_DIR, "lstm_model.png"))
plt.close()

