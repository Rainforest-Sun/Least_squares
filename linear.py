# Author 咕咕队摸大鱼

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv  # 矩阵求逆
from numpy import dot, multiply, double  # 矩阵点乘
from numpy import mat  # 二维矩阵

def Linear_least_squares(X,Y):
    w = dot(dot(inv(dot(X.T, X)), X.T), Y)
    return w

def Gradient_descent(X,Y,m):
    w = np.random.rand(m).reshape(m, 1)  # 初始值
    alpha = 0.01  # 学习率
    for i in range(500000):
        w = w + alpha * dot(X.T,(Y - dot(X, w)))
    return w

def Newton_method(X,Y,m):
    w = np.random.rand(m).reshape(m, 1)  # 初始值
    for i in range(10000):
        H = np.array([X.T*X.T[k].T for k in range(m)]).reshape(m,m) # Hessian矩阵
        w = w + dot(inv(H) , dot(X.T,(Y - dot(X, w))))
    return w

def Gauss_Newton_method(X,Y,m):
    w = np.random.rand(m).reshape(m, 1)  # 初始值
    for i in range(10000):
        Jr = -X # Jacobian矩阵
        w = w - dot(inv(dot(Jr.T,Jr)) , dot(Jr.T,(Y - dot(X, w))))
    return w

if __name__=="__main__":
    n = 101
    X = np.linspace(0,10,n)
    print(X)
    noise = np.random.randn(n)
    Y = 2.5 * X + 0.8 + noise
    X=mat([X,[1 for i in range(n)]]).T
    Y=mat(Y).T
    print(Y)
    # w=Linear_least_squares(X,Y)
    # print(w)
    # w=Gradient_descent(X,Y)
    # print(w)
    w=Newton_method(X,Y,2)
    print(w)
    # w=Gauss_Newton_method(X,Y)
    # print(w)
    plt.scatter(X.T[0].T.tolist(),Y.tolist())
    plt.plot(np.arange(0,10,1), double(w[0])*np.arange(0,10,1)+double(w[1]), color = 'r', marker = 'o', label = 'Y = X * w')
    plt.savefig('plt-散点图1.jpg')