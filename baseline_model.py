import os
import settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings


# Baseline model
def baseline_model(train, test):
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # make prediction
        yhat = history[-1]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # report performance   
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse, predictions

'''
# load data
series = pd.Series.from_csv(os.path.join(settings.PROCESSED_DIR, 'train.csv'))
# prepare data
# split into train and validation sets
X = series.values.astype('float32')
train, validation = X[0:-12], X[-12:]

# fit model
rmse, predictions = baseline_model(train, validation)

print('Baseline model:')
print('Validation RMSE: %.3f' % rmse)
'''


# load data
train = pd.Series.from_csv(os.path.join(settings.PROCESSED_DIR, 'train.csv'))
test = pd.Series.from_csv(settings.PROCESSED_DIR + 'test.csv')
# prepare data
X = train.values.astype('float32')
y = test.values.astype('float32')

# fit model
rmse, predictions = baseline_model(X, y)

print('Baseline model:')
print('Test RMSE: %.3f' % rmse)

'''
# plot predictions and expected results
plt.plot(X)
plt.plot([None for i in X] + [x for x in Y])
plt.plot([None for i in X] + [x for x in predictions])
plt.show()
'''

plt.plot(y)
plt.plot(predictions, color='red')
plt.title('Persistence Model for test set')
plt.savefig(os.path.join(settings.OUTPUT_DIR, "persistence_model.png"))
plt.close()
