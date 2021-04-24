import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression as LR
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import StandardScaler as SS
from statsmodels.api import add_constant


births = pd.read_csv('birth.txt')
births['time'] = np.arange(len(births))+1

births.index = pd.date_range('1980-01','2010-12', freq = 'MS')
births['year'] = births.index.year
births['month'] = (births.index.month-1)//3

births = pd.concat([births, pd.get_dummies(births['month'], prefix='month')], axis = 1)


# future = pd.DataFrame({'time':np.arange(births['time'].max()+1, births['time'].max()+13)})

divider = 1
future = births[['time']].iloc[-divider:].copy()

_2011 = pd.DataFrame({'time': np.arange(future['time'].max()+1, future['time'].max()+25)})
_2011.index = pd.date_range(future.index.max()+pd.DateOffset(months=1), periods = len(_2011), freq = 'MS')

future = pd.concat([future, _2011])

original = births.copy()
births = births.iloc[:-divider,:].copy()

# future.index = pd.date_range('2011-1', '2011-12', freq = 'MS')
future['month'] = (future.index.month-1)//3
future = pd.concat([future, pd.get_dummies(future['month'], prefix='month')], axis = 1)

# for i in ['year','month']:
#     plt.plot(births.groupby(i)['num_births'].mean())
#     plt.show()
    

# plt.plot(births.loc['2006':'2011', 'num_births'])
# plt.show()


plt.plot(births.num_births)
plt.plot(births.resample('Y')['num_births'].mean())
plt.plot(births.resample('Q-NOV')['num_births'].mean())
plt.show()


bics, aics = [], []

upper = 6

for i in range(2, upper):
    births[f'time^{i}'] = births['time']**i
    future[f'time^{i}'] = future['time']**i

    columns = [c for c in births.columns if 'time' in c or 'month_' in c] 

    ss = SS()

    X = ss.fit_transform(births[columns].values.reshape(-1, len(columns)))
    X = add_constant(X)

    future_X = add_constant(ss.transform(future[columns]))
    
    # X = SS().fit_transform(births[columns].values.reshape(-1, len(columns)))

    # model1 = LR().fit(births[columns].values.reshape(-1,len(columns)), births['num_births'])
    model2 = OLS(births['num_births'], X).fit()
    
    bics.append(model2.bic)
    aics.append(model2.aic)

    plt.plot(births.index, 
                model2.predict(X), 
                label = f'{i}:{model2.bic:.3f}')

plt.plot(future.index, model2.predict(future_X), marker = '*', color = 'k', linestyle = '-')

plt.legend()
plt.plot(original['num_births'], marker = '*', linestyle = '')
plt.show()

plt.plot(range(2,upper), aics)
plt.plot(range(2,upper), bics)
plt.axvline(2+np.argmin(bics))
plt.axvline(2+np.argmin(aics))

plt.show()

residuals = model2.predict(X) - births['num_births']
plt.plot(births.index, residuals)
plt.axhline(0)
plt.show()