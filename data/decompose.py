import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression as LR
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import StandardScaler as SS
from statsmodels.api import add_constant


class Decomposer:

    def __init__(self):
        self.births = pd.read_csv('birth.txt')
        self.births.index = pd.date_range('1980-01','2010-12', freq = 'MS')


    def estimate_trend(self):
        lr = LR()
        self._X = (self.births.index - self.births.index.min()).days.values.reshape(-1,1)
        lr.fit(self._X, self.births['num_births'])
        self.lr = lr

    def detrend(self, plot = False):
        if 'lr' not in self.__dict__:
            self.estimate_trend()

        trend = self.lr.predict(self._X)
        self.births['detrended'] = self.births['num_births'] - trend

        if plot:
            plt.plot(self.births['num_births'])
            plt.axhline(self.births['num_births'].mean())
            plt.plot(self.births['detrended'])
            plt.axhline(self.births['detrended'].mean())
            plt.show()


    def compute_moving_average(self, window = 12, plot = False):
        if 'detrended' not in self.births.columns:
            self.detrend()

        MA = self.births['detrended'].rolling(window, center=True).mean()
        

        first = MA[~MA.isna()].iloc[0]
        last = MA[~MA.isna()].iloc[-1]

        MA[:window].fillna(first, inplace = True)
        MA[-window:].fillna(last, inplace = True)


        self.births['MA'] = MA
        
        # self.births['MA'].fillna(MA.mean(), inplace=True)
        if plot:
            plt.plot(self.births['MA'])
            plt.plot(self.births['detrended'])
            plt.plot(self.births['MA'] - self.births['detrended'])
            plt.show()

    def estimate_seasonality(self):
        pass

    def estimate_cyclical_terms(self):

        pass

    def __repr__(self):
        return str(self.births)


if __name__ == '__main__':
    d = Decomposer()
    print(d)

    # d.estimate_trend()