import numpy as np
import scipy as sp
import math
import random
from scipy.optimize.slsqp import approx_jacobian
import matplotlib.pyplot as plt

class GaussianKernel(object):
    # カーネル関数のパラメータa,bを初期化
    def __init__(self, params):
        assert np.shape(params) == (2,)
        self.__params = params

    # カーネル関数のパラメータa,bを返す
    def get_params(self):
        return np.copy(self.__params)

     # x,yを入力としてカーネル関数の値を計算 PRML式(6.63)
    def __call__(self, x, y):
        return (self.__params[0])* np.exp(-0.5 * self.__params[1] * (x - y) ** 2)

    # x,yを入力とした時のカーネル関数のパラメータでの微分を計算
    def derivatives(self, x, y):
        sq_diff = (x - y) ** 2
        # パラメータaでの微分
        delta_0 = 2*self.__params[0]*np.exp(-0.5 * self.__params[1] * sq_diff)
        # パラメータbでの微分
        delta_1 = -0.5 * sq_diff * delta_0 * self.__params[0]**2
        return (delta_0, delta_1)

    # カーネル関数のパラメータを更新
    def update_parameters(self, updates):
        assert np.shape(updates) == (2,)
        self.__params += updates

