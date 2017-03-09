import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import settings
import os

series = pd.Series.from_csv(os.path.join(settings.DATA_DIR, 'Average-number-of-visitors-in-New-Zealand-for-business.csv'), header=0)

# create dataset.csv and test.csv
split_point = len(series) - 5
dataset, test = series[0:split_point], series[split_point:]
print('dataset %d, Test %d' % (len(dataset), len(test)))
dataset.to_csv(settings.PROCESSED_DIR + 'dataset.csv')
test.to_csv(settings.PROCESSED_DIR + 'test.csv')


print(dataset.describe())

# line plot
plt.plot(dataset)
plt.show()


# check if it is random walk

# autocorrelation plot
# The time series shows a strong temporal dependence that
# decays linearly or in a similar pattern.
autocorrelation_plot(dataset)
plt.show()

# Histograms and density plots provide insight into the distribution of all observations
# histograms
plt.hist(dataset)
plt.show()
# density plot
dataset.plot(kind='kde')
plt.show()

# decomposition plot 
decomposition = seasonal_decompose(dataset)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(dataset, label='Original')
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


# Box and Whisker Plots
groups = dataset['1999':'2011'].groupby(pd.TimeGrouper('A'))
years = pd.DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.boxplot()
plt.show()
