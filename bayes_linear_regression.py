import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
import pandas as pd
from itertools import chain
import random
import scipy as sp
from scipy.optimize import leastsq


def array(x):
 #计算得到设计矩阵
 #input ： 数据矩阵
 #返回设计矩阵
 PHI =[]
 for i in x:
  PHI.append(phi(i))
 PHI = np.array(PHI)
 return PHI


def phi(x) :
 #基函数
 #基函数是幂函数最高次是7次方
 phi = list([])
 for i in range(8):
   phi.append(pow(x,i))
 return phi
#开始用高斯分布，但是拟合效果不是很好，就果断改用幂函数


alpha_martix=0
beta_martix =0
#迭代计算alpha，beta
def iteration(x,t,alpha,beta,PHI):
 while(1):
    alpha_iteration = alpha
    beta_iteration = beta
    yeta = 0.0
    sum = 0.0

    Sigma_N = np.linalg.inv(alpha * np.identity(PHI.shape[1]) + beta * np.dot(PHI.T, PHI))
    mu_N = beta* np.dot(Sigma_N, np.dot(PHI.T, t))
    lamda, b= np.linalg.eig(beta * np.dot(PHI.T, PHI))#计算出特征值

    for lam in lamda:
       yeta = yeta +float(lam / (alpha + lam))
    alpha = yeta / (np.dot(mu_N.T, mu_N))

    for i in range(t.shape[0]):
       sum += (t[i] - np.dot(PHI[i],mu_N) ** 2 )
    beta = (t.shape[0] - yeta) / sum
    beta = 1 / beta

    if((abs(alpha_iteration-alpha )<0.001) and (abs(beta_iteration - beta) < 0.001)):
        return alpha, beta
        break


def rmse(y_test, y):
 #均方根误差
 fsum = 0.0
 for i in range(len(y)):
     fsum += pow(y_test[i] - y[i],2)
 return np.sqrt(fsum / len(y))

if __name__ == '__main__':
 #固定alpha，beta的贝叶斯线性回归和迭代计算出alpha，beta贝叶斯回归
 #
 #导入数据
 #分别提取出两列

 filename = 'data.csv'
 names = ['data', 'label']
 data = read_csv(filename, names=names, skiprows=1)
 data_name = ['data']
 X = pd.DataFrame(data, columns=data_name)
 X = np.array(X)
 label_name = ['label']
 t = pd.DataFrame(data, columns=label_name)
 t = np.array(t)
 #随机分配训练集和测试集
 x_train,y_train,x_vali,y_vali = train_test_split(X,t,test_size=0.7)

 #计算设计矩阵，用于贝叶斯回归
 PHI = array(x_train)
 PHI = np.reshape(PHI, (-1, 8))


 alpha = 0.1 # alpha初始值
 beta = 9.0  # beta初始值

 Sigma_N = np.linalg.inv(alpha* np.identity(PHI.shape[1]) + beta* np.dot(PHI.T, PHI))#PRML第153页公式3.53
 mu_N = beta * np.dot(Sigma_N, np.dot(PHI.T, x_vali))#PRML第153页公式3.54

 alpha_iteration,beta_iteration = iteration(x_train,x_vali,alpha,beta,PHI)#通过迭代计算出先验的超参
 Sigma_N0 = np.linalg.inv(alpha_iteration * np.identity(PHI.shape[1]) + beta_iteration * np.dot(PHI.T, PHI))#计算出贝叶斯回归的Sn
 mu_N0 = beta_iteration * np.dot(Sigma_N0, np.dot(PHI.T, x_vali))#计算出贝叶斯回归的Mn

 # 画图
 xlist = np.arange(0,13,0.1)
 mean = np.dot( array(xlist),mu_N0)
 mean0 = np.dot(array(xlist), mu_N)

 plt.plot(xlist, mean, 'g')
 plt.plot(xlist, mean0, 'r')

 plt.plot(X, t, 'o')
 plt.show()

 #测试部分
 PHI1 = array(y_train)
 PHI1 = np.reshape(PHI1, (-1, 8))
 alpha_iteration_test,beta_iteration_test = iteration(y_train,y_vali,alpha,beta,PHI1)
 Sigma_N1 = np.linalg.inv(alpha_iteration_test * np.identity(PHI.shape[1]) + beta_iteration_test * np.dot(PHI1.T, PHI1))
 mu_N1 = beta_iteration_test * np.dot(Sigma_N1, np.dot(PHI1.T, y_vali))
 mean1 = np.dot(PHI1, mu_N1)
 print(rmse(y_vali,mean1 ))

