import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from scipy.stats import norm

#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from keras.utils import np_utils
import matplotlib.mlab as mlab

df = pd.read_csv('data/Model_Input_Data/_data_final_v2_Q_returns.csv', sep=',')
df = df.fillna(0)
X = df.iloc[:, 7:-1]
"""
fields = ['Book/Market', 'Enterprise Value Multiple', 'P/E (Diluted, Incl. EI)', 'Price/Sales', \
          'Price/Cash flow', 'Dividend Payout Ratio', 'Net Profit Margin', 'Gross Profit Margin', \
          'Cash Flow Margin', 'Return on Assets', 'Return on Equity', 'Return on Capital Employed', \
          'Gross Profit/Total Assets', 'Total Debt/Invested Capital', 'Inventory/Current Assets', \
          'Total Debt/Total Assets', 'Cash Ratio', 'Quick Ratio (Acid Test)', 'Current Ratio', 'Inventory Turnover', \
          'Asset Turnover', 'Price/Book', 'Dividend Yield', 'Volume Change (3mo)', \
          'Change in Shares Outstanding (3mo)', 'Total Volatility']
X = X[fields]
"""
yT = df.iloc[:, -1]
n = len(yT)
y = pd.Series([])
k = 523
inf = 100
for i in range(0, int(n/k)):
    vals = yT[i*k:(i+1)*k]
    mu, std = norm.fit(vals)
    #bins = [-inf, mu - 2*std, mu + 2*std, inf]
    #bins = [-inf, mu - 2 * std, mu - std, mu, mu + std, mu + 2 * std, inf]
    #yCat = pd.cut(vals, bins=bins, labels=False)
    yCat = pd.qcut(vals, 4, labels=False)
    y = pd.concat([y, yCat])


X_trainDev, X_test, y_trainDev, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_dev, y_train, y_dev = train_test_split(X_trainDev, y_trainDev, test_size=0.25, random_state=1)

lm = LinearRegression()
lm.fit(X_train, y_train)

pred_train = lm.predict(X_train)
pred_train = np.round(pred_train)
pred_dev = lm.predict(X_dev)
pred_dev = np.round(pred_dev)

print("Training Error MSE: " + str(mean_squared_error(y_train, pred_train)))
print("Dev Error MSE: " + str(mean_squared_error(y_dev, pred_dev)))
print("Training accuracy: " + str(accuracy_score(y_train, pred_train)))
print("Dev accuracy: " + str(accuracy_score(y_dev, pred_dev)))
print("Dev report: " + str(classification_report(np.array(y_dev), pred_dev)))

y_dev = np.array(y_dev).reshape(-1, 1)

mu, std = norm.fit(y_train)
y_rand = np.random.normal(loc=mu, scale=std, size=(len(y_dev), 1))
y_rand = np.round(y_rand)

print("Random baseline.....")
print("Dev error (random sampling): " + str(mean_squared_error(y_dev, y_rand)))
print("Dev accuracy (random sampling): " + str(accuracy_score(y_dev, y_rand)))
print("Dev report (random sampling): " + str(classification_report(np.array(y_dev), y_rand)))

plt.figure(1)
plt.hist(y_train, normed=True)
plt.xlim(min(y_train), max(y_train))
x = np.linspace(min(y_train), max(y_train), 100)
plt.plot(x, mlab.normpdf(x, mu, std))
plt.show()


#hist = df.hist(column='Returns (1 year Fwd)', bins=300)
#plt.show()
