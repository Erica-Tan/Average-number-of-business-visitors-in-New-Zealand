import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import settings
import os

df_data = pd.read_csv(os.path.join(settings.DATA_DIR, 'Average-number-of-visitors-in-New-Zealand-for-business.csv'), header=0)
dates = pd.to_datetime(df_data["Date"], format = '%YM%m')
df_data["Date"] = dates.apply(lambda x: x.strftime('%Y-%m-%d'))
df_data.to_csv(os.path.join(settings.PROCESSED_DIR, "dataset.csv"), index=False)



series = pd.Series.from_csv(os.path.join(settings.PROCESSED_DIR, 'dataset.csv'), header=0)


# create train.csv and test.csv
split_point = len(series) - 5
train, test = series[0:split_point], series[split_point:]
print('Train %d, Test %d' % (len(train), len(test)))
train.to_csv(settings.PROCESSED_DIR + 'train.csv')
test.to_csv(settings.PROCESSED_DIR + 'test.csv')


print(train.describe())

# line plot
plt.plot(train)
plt.show()


# check if it is random walk

# autocorrelation plot
# The time series shows a strong temporal dependence that
# decays linearly or in a similar pattern.
autocorrelation_plot(train)
plt.show()

# Histograms and density plots provide insight into the distribution of all observations
# histograms
plt.hist(train)
plt.show()
# density plot
train.plot(kind='kde')
plt.show()

# decomposition plot 
decomposition = seasonal_decompose(train)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(train, label='Original')
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


# Box and Whsker Plots
groups = train['1999':'2011'].groupby(pd.TimeGrouper('A'))
years = pd.DataFrame()
for name, group in groups:
	years[name.year] = group.values
years.boxplot()
plt.show()

