import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings

series = pd.Series.from_csv('data/Average-number-of-visitors-in-New-Zealand-for-business.csv', header=0)


print(series.head())

print('\n Data Types:')
print(series.dtypes)

plt.plot(series)
plt.show()


# check if it is random walk

# autocorrelation plot
# The time series shows a strong temporal dependence that
# decays linearly or in a similar pattern.
autocorrelation_plot(series)
plt.show()

# Histograms and density plots provide insight into the distribution of all observations
# histograms
plt.hist(series)
plt.show()
# density plot
series.plot(kind='kde')
plt.show()

# decomposition plot 
decomposition = seasonal_decompose(series)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(series, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# check if it is stationary
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12,center=False).mean()
    rolstd = timeseries.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


test_stationarity(series)


# make it stationary
ts_diff = series - series.shift()
ts_diff.dropna(inplace=True)

plt.plot(ts_diff)
plt.show()


test_stationarity(ts_diff)


# Baseline model
# Create lagged dataset
values = pd.DataFrame(series.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']


# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.9)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]



# persistence model
def model_persistence(x):
	return x

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Baseline model:')
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y])
plt.plot([None for i in train_y] + [x for x in predictions])
plt.show()


# Grid search ARIMA model
lag_acf = acf(ts_diff, nlags=40)
lag_pacf = pacf(ts_diff, nlags=40, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()



# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# split into train and test sets
	train_size = int(len(X) * 0.60)
	train, test = X[0:train_size], X[train_size:]

	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
		
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None

	# Iterate ARIMA Parameters
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

p_values = range(0, 4)
d_values = [1]
q_values = range(0, 4)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)


# ARIMA model

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(train, test,  arima_order):
        # walk forward over time steps in test
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0][0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
		
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)
	print('RSS: %.4f'% sum((predictions-test)**2))

	# plot
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.title('SME: %.4f'% error)
	plt.show()

	return predictions


warnings.filterwarnings("ignore")
dataset = series.astype('float')
dataset = dataset.values

# split into train and test sets
train_size = int(len(dataset) * 0.60)
train, test = dataset[0:train_size], dataset[train_size:]

predictions = evaluate_arima_model(train, test, (3, 1, 2))


# calculate residuals
residuals = [test[i]-predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)

# plot residuals
residuals.plot()
plt.show()

# summary statistics
print(residuals.describe())

# histogram plot
residuals.hist()
plt.show()

# density plot
residuals.plot(kind='kde')
plt.show()
