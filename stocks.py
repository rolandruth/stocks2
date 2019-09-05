import pandas as pd
import math
import numpy as np

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

# load in tesla data
data = pd.read_csv("TSLA.csv")
data.head()

dfreg = data.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
dfreg['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
dfreg.head()

dfreg.fillna(value=-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(dfreg)))

# AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(dfreg['label'])
y = y[:-forecast_out]

# set up test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
          metric_params=None, n_jobs=1, n_neighbors=2, p=2,
          weights='uniform')

# get confidence values
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

#########################################
# forecast Linear regression
#########################################
forecast_set_reg = clfreg.predict(X_lately)

#########################################
# forecast quadratic regression 2
#########################################
forecast_set_poly2 = clfpoly2.predict(X_lately)

#########################################
# forecast quadratic regression 3
#########################################
forecast_set_poly3 = clfpoly3.predict(X_lately)

#########################################
# forecast knn regression
#########################################
forecast_set_knn = clfknn.predict(X_lately)

#########################################
# best performing one by price
#########################################
stock_prices = [forecast_set_reg[0], forecast_set_poly2[0], forecast_set_poly3[0], forecast_set_knn[0]]
regression_type = ["Linear Regression", "Quadratic Regression 2", "Quadratic Regression 3", "KNN Regression"]

top_price = 0
top_regression = ""
counter = 0

# figure out top price
for i in stock_prices:
    if stock_prices[counter] > top_price:
        top_price = stock_prices[counter]
        top_regression = regression_type[counter]
        top_counter = counter
    counter += 1

# print out the best performing one
print("Best preforming type is "+top_regression+" with a price of "+str(top_price))

#########################################
# display chart of best preforming one
#########################################

dfreg['Forecast'] = np.nan
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + 1

# determine which graph to display
if top_counter == 0:
    for i in forecast_set_reg:
        next_date = next_unix
        next_unix += 1
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
elif top_counter == 1:
    for i in forecast_set_poly2:
        next_date = next_unix
        next_unix += 1
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
elif top_counter == 2:
    for i in forecast_set_poly3:
        next_date = next_unix
        next_unix += 1
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
elif top_counter == 3:
    for i in forecast_set_knn:
        next_date = next_unix
        next_unix += 1
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]

dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
