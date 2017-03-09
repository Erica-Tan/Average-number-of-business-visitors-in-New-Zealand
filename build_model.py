import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from scipy.stats import boxcox
from math import sqrt
from math import log
from math import exp
import warnings
import settings
import os

# load data
series = pd.Series.from_csv(os.path.join(settings.PROCESSED_DIR, 'dataset.csv'))
# prepare data
# split into train and validation sets
X = series.values.astype('float32')
train_size = int(len(X) * 0.60)
train, validation = X[0:train_size], X[train_size:]

# Baseline model
def baseline_model(train, test):
        # walk-forward validation
        history = [x for x in train]
        predictions = list()
        for i in range(len(test)):
            # predict
            yhat = history[-1]
            predictions.append(yhat)
            
            # observation
            obs = validation[i]
            history.append(obs)
        mse = mean_squared_error(test, predictions)
        rmse = sqrt(mse)
        print('Baseline model:')
        print('Test RMSE: %.3f' % rmse)

        # plot predictions and expected results
        plt.plot(train)
        plt.plot([None for i in train] + [x for x in test])
        plt.plot([None for i in train] + [x for x in predictions])
        plt.show()

        return predictions

baseline_model(train, validation)


# ARIMA Model
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


'''
# Grid Search ARIMA model
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(train, test, arima_order):
	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)

		obs = test[t]
		history.append(obs)
		#print('predicted=%f, expected=%f' % (yhat, obs))
		
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(train, test, p_values, d_values, q_values):
	best_score, best_cfg = float("inf"), None

	# Iterate ARIMA Parameters
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(train, test, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

p_values = range(0, 4)
d_values = [1]
q_values = range(0, 4)
warnings.filterwarnings("ignore")
evaluate_models(train, validation, p_values, d_values, q_values)
'''

# ARIMA model
# an ARIMA model for a given order (p,d,q)
def arima_model(train, test,  arima_order):
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
		#print('predicted=%f, expected=%f' % (yhat, obs))
		
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
    
	# plot
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.title('RSME: %.4f'% rmse)
	plt.show()

	return predictions

warnings.filterwarnings("ignore")
predictions = arima_model(train, validation, (1, 1, 2))



# calculate residuals
residuals = [validation[i]-predictions[i] for i in range(len(predictions))]
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

'''
# Grid Search Transform ARIMA model
# invert box-cox transform
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)


# evaluate an ARIMA model for a given order (p,d,q)
def transform_arima_model(train, test, arima_order):
	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		# transform
		transformed, lam = boxcox(history)
		if lam < -5:
		    transformed, lam = history, 1
					
		model = ARIMA(transformed, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]

		# invert transformed prediction
		yhat = boxcox_inverse(yhat, lam)
		
		predictions.append(yhat)

		obs = test[t]
		history.append(obs)

		#print('predicted=%f, expected=%f' % (yhat, obs))
		
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
	return rmse, predictions

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(train, test, p_values, d_values, q_values):
	best_score, best_cfg = float("inf"), None

	# Iterate ARIMA Parameters
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = transform_arima_model(train, test, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

p_values = range(0, 4)
d_values = [1]
q_values = range(0, 4)
warnings.filterwarnings("ignore")
evaluate_models(train, validation, p_values, d_values, q_values)
'''

# Transform ARIMA model
# invert box-cox transform
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)


# an transform ARIMA model for a given order (p,d,q)
def transform_arima_model(train, test, arima_order):
	# walk-forward validation
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		# transform
		transformed, lam = boxcox(history)
		if lam < -5:
		    transformed, lam = history, 1
					
		model = ARIMA(transformed, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]

		# invert transformed prediction
		yhat = boxcox_inverse(yhat, lam)
		
		predictions.append(yhat)

		obs = test[t]
		history.append(obs)

		#print('predicted=%f, expected=%f' % (yhat, obs))
		
	# calculate out of sample error
	mse = mean_squared_error(test, predictions)
	rmse = sqrt(mse)
    
	# plot
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.title('RSME: %.4f'% rmse)
	plt.show()

	return predictions

warnings.filterwarnings("ignore")
predictions = transform_arima_model(train, validation, (1, 1, 2))


# calculate residuals
residuals = [validation[i]-predictions[i] for i in range(len(predictions))]
residuals = pd.DataFrame(residuals)

# summary statistics
print(residuals.describe())

# density plot
residuals.plot(kind='kde')
plt.show()


# Finalize Model
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
 
ARIMA.__getnewargs__ = __getnewargs__

# load data
dataset = pd.Series.from_csv(settings.PROCESSED_DIR + 'dataset.csv')
# prepare data
X = dataset.values.astype('float32')
# transform data
warnings.filterwarnings("ignore")
transformed, lam = boxcox(X)
# fit model
model = ARIMA(transformed, order=(1, 1, 2))
model_fit = model.fit(disp=0)
# save model
model_fit.save(settings.OUTPUT_DIR + 'model.pkl')
np.save(settings.OUTPUT_DIR + 'model_lambda.npy', [lam])


