import os
import settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings


# load data
series = pd.Series.from_csv(os.path.join(settings.PROCESSED_DIR, 'train.csv'))
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
        # make prediction
        yhat = history[-1]
        predictions.append(yhat)
        # observation
        obs = test[i]
        history.append(obs)
    # report performance   
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse, predictions

rmse, predictions = baseline_model(train, validation)

print('Baseline model:')
print('Validation RMSE: %.3f' % rmse)

# plot predictions and expected results
plt.plot(train)
plt.plot([None for i in train] + [x for x in test])
plt.plot([None for i in train] + [x for x in predictions])
plt.show()
