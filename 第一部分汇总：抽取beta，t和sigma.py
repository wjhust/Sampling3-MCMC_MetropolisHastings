# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:59:10 2018

@author: admin
"""
import random
import math as m
import numpy as np


#gama函数
from scipy.special import gamma
#变下限gamma函数
from scipy.special import gammaincc
random.seed(10)

# x是nxp矩阵，y是nx1矩阵，这里设n=2
p=5
n=2
x=np.ones((2,5))
#t=[2,3,4,5,6]
y=np.ones((2,1))
lambda2=0.2
lambda1=0.2
#sigma=1

#这里x是2X5矩阵，y是2X1矩阵，beta需要是5X1矩阵


#=========================================多维高斯分布的抽样==================================================
def gauss_beta(y,x,t,lambda2,sigma,p):
    diag=np.zeros(shape=(p,p))
    for i in range(len(diag)):
        diag[i][i]=(t[i]/(t[i]-1))
        matrixA=np.mat(np.dot(np.transpose(x),x) + lambda2 * diag)
    s=np.dot(matrixA.I,np.transpose(x))
    mean=np.asarray(np.transpose(np.dot(s,y))) # 1x5
    s=[mean[0][i] for i in range(len(mean[0]))]
    var=(sigma * sigma) * matrixA.I # 5x5
    #要求mean是向量,1维，但这里mean是矩阵1X5,需要不停的转换格式
    s=np.random.multivariate_normal(s,var)
    #不加下边的格式转化，那么输出的格式是numpy.ndarray，即不带逗号，因此不一定符合后边的输入要求
    res=[]
    for i in s:
        res.append(i)
    #输出的是list，转化为5X1矩阵
    return np.transpose(np.mat(res))

#beta=np.transpose(np.mat(gauss_beta(y,x,t,lambda2,sigma,p)))   
#print(beta.shape)
'''
测试
#print(gauss_beta(y,x,t,lambda2,sigma,p))
beta是5X1
'''

#=================================================逆高斯分布GIG==============================================================
def ig_t(beta,sigma,lambda1,lambda2):
    b1=np.asarray(beta)
    s4=[]
    for i in range(len(b1)):
        b2=b1[i]
        s4.append(b2[0])  
    ans=[]
    scale=lambda1/(4*lambda2*m.pow(sigma,2))
    for i in range(len(s4)):
        mean=m.sqrt(lambda1)/(2*lambda2*abs(s4[i]))
        res=(1 + 1/(np.random.wald(mean,scale,size=1))[0])
        ans.append(res)
        
    return ans

#==================================================拒绝接受抽样抽取sigma^2=====================================================
    
def ac_re_sigma(y,x,t,beta,lambda1,lambda2,n,p):
    a=n/2+p
    #使用np.dot之后需要转化数据格式，asarray+
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
    return sigma[0]
#抽出来的是一个数字
'''
测试
print(ac_re_sigma(y,x,t,gauss_beta(y,x,t,lambda2,sigma,p),lambda1,lambda2,n,p))
'''

'''
整体测试
beta=gauss_beta(y,x,t,lambda2,sigma,p)
sigma2=ac_re_sigma(y,x,t,beta,lambda1,lambda2,n,p)
t=ig_t(beta,sigma2,lambda1,lambda2)
print("beta:",beta,"sigma:",sigma2,"t:",t)
'''
#==============================gibbs循环采样===========================================================
def gibbs(N_hops):
    beta_ans=[]
    sigma_ans=[1]
    t_ans=[[2,3,4,5,6]]
    for i in range(N_hops):
        beta_ans.append(gauss_beta(y,x,t_ans[-1],lambda2,sigma_ans[-1],p))
        sigma_ans.append(ac_re_sigma(y,x,t_ans[-1],beta_ans[-1],lambda1,lambda2,n,p))
        t_ans.append(ig_t(beta_ans[-1],sigma_ans[-1],lambda1,lambda2))
    print("参数beta的估计为：",beta_ans[-1])
    print("参数sigma^2的估计为：",sigma_ans[-1])
    print("参数t的估计为",t_ans[-1])

gibbs(1)
