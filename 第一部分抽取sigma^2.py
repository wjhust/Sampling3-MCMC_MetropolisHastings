# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:37:37 2018

@author: admin
"""
import numpy as np
#gama函数
from scipy.special import gamma
#变下限gamma函数
from scipy.special import gammaincc
import random
import math as m

#输入的y,x,beta，t是矩阵，其他为参数
p=5
n=2
x=np.ones((2,5))
t=[2,3,4,5,6]
y=np.ones((2,1))
lambda2=2
lambda1=2
sigma=1
beta=np.transpose(np.mat([0.06164318966243999, -0.4960005183092272, 0.8145349578205902, -0.35177053292771066, 1.0112485803148172]))
def ac_re_sigma(y,x,t,beta,lambda1,lambda2,n,p):
    a=n/2+p
    m1=np.dot(np.transpose(y-np.dot(x,beta)),(y-np.dot(x,beta)))
    b1=np.asarray(m1)
    b2=[b1[0][i] for i in range(len(b1[0]))]
    b3=b2[0]
    t1=0
    t2=0
    for i in range(len(t)):
        t1+=t[i]/(t[i]-1) * m.pow(beta[i],2) * lambda2
        t2+=t[i] * (m.pow(lambda1,2))/(4*lambda2)
    b=(b3+t1+t2)/2
    #从逆gamam(a,b)分布中抽取候选
    z=np.random.wald(a,b)
    u=random.uniform(0,1)
    sigma=[]
    #需要搞清楚论文中此处公式的gama参数的问题
    s=p*m.log(gamma(1/2)) - p * m.log(gamma(1/2))*gammaincc(1/2,m.pow(lambda1,2)/(8 * z *lambda2))    
    if m.log(u)<s:
        sigma.append(z)
    return sigma
    
print(ac_re_sigma(y,x,t,beta,lambda1,lambda2,n,p))
    
    
        
