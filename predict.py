import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from scipy.stats import boxcox
from math import sqrt
from math import log
from math import exp
import settings
import os


# Transform ARIMA model
# invert box-cox transform
def boxcox_inverse(value, lam):
	if lam == 0:
		return exp(value)
	return exp(log(lam * value + 1) / lam)


# Validate Model
# load and prepare datasets
dataset = pd.Series.from_csv(settings.PROCESSED_DIR + 'dataset.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
test = pd.Series.from_csv(settings.PROCESSED_DIR + 'test.csv')
Y = test.values.astype('float32')

# load model
model_fit = ARIMAResults.load(os.path.join(settings.OUTPUT_DIR, 'model.pkl'))
lam = np.load(os.path.join(settings.OUTPUT_DIR, 'model_lambda.npy'))

warnings.filterwarnings("ignore")

# make first prediction
predictions = list()
yhat = model_fit.forecast()[0]
yhat = boxcox_inverse(yhat, lam)
predictions.append(yhat)
history.append(Y[0])
#print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

# rolling forecasts
for i in range(1, len(Y)):
	# transform
	transformed, lam = boxcox(history)
	if lam < -5:
		transformed, lam = history, 1

	# predict
	model = ARIMA(transformed, order=(1, 1, 2))
	model_fit = model.fit(disp=0)
	yhat = model_fit.forecast()[0]

	# invert transformed prediction
	yhat = boxcox_inverse(yhat, lam)

	predictions.append(yhat)
	# observation
	obs = Y[i]
	history.append(obs)
	print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(Y, predictions)
rmse = sqrt(mse)

plt.plot(Y)
plt.plot(predictions, color='red')
plt.title('RSME: %.4f'% rmse)
plt.show()
