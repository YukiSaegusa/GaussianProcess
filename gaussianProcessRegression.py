
import numpy as np
import scipy as sp
import math
import random
from scipy.optimize.slsqp import approx_jacobian
import matplotlib.pyplot as plt

class GaussianProcessRegression(object):
    # カーネル関数とノイズの精度パラメータの初期化
    def __init__(self, kernel, beta=1.):
        self.kernel = kernel
        self.beta = beta #論文ではγ

    # カーネル関数のパラメータ推定を行わずに回帰
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.K = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(len(x)):
                self.K[i][j] = self.kernel(x[i],x[j])

    # カーネル関数のパラメータの推定を行う回帰
    def fit_kernel(self, x, t, learning_rate=0.1, iter_max=10000):
        for i in range(iter_max):
            params = self.kernel.get_params()
            # カーネル関数の今のパラメータで回帰
            self.fit(x, t)
            # 対数エビデンス関数をパラメータで微分
            gradients = self.kernel.derivatives(*np.meshgrid(x, x))
            # パラメータの更新量を計算 PRML式(6.70)
            updates = np.array(
                [-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad in gradients])
            # パラメータを更新
            self.kernel.update_parameters(learning_rate * updates)
            # パラメータの更新量が小さければ更新をやめる
            
            if np.allclose(params, self.kernel.get_params()):
                break
        else:
            # 既定の更新回数だけ更新してもパラメータの更新量が小さくない場合以下の文を出力
            print("parameters may not have converged")
            
    def kernels(self,x):
        return np.array([self.kernel(x,_x) for _x in Xt]).T
    
    def kernels_dif(self,x):
        params = self.kernel.get_params()
        return np.array([- params[1]*(self.x[i]-x)*params[0]**2 *np.exp(-0.5*params[1]*(self.x[i]-x)**2) for i in range(len(self.x))]).T
    
    # 予測分布を出力
    def predict_dist(self, x):
        K = self.kernel(*np.meshgrid(x, self.x, indexing='ij'))
        # 予測分布の平均を計算 PRML式(6.66)
        mean = K.dot(self.precision).dot(self.t)
        # 予測分布の分散を計算 PRML式(6.67)
        var = self.kernel(x, x) + 1 / self.beta - np.sum(K.dot(self.precision) * K, axis=1)
        return mean.ravel(), np.sqrt(var.ravel())
    
    def mean(self,x):
        return self.kernels(x).T.dot(np.linalg.inv(self.K + self.beta*np.eye(len(self.x)))).dot(self.y)
    
    def mean_j(self,x):
        return self.kernels_dif(x).T.dot(np.linalg.inv(self.K + self.beta*np.eye(len(self.x)))).dot(self.y)
    
    def var(self,x):
        return self.kernel(x,x)- self.kernels(x).T.dot((np.linalg.inv(self.K)).dot(self.kernels(x)))

