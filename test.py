# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:28:14 2018

@author: admin
"""

import math as m
import random

def houyan1(lambda1,beta,alafa1,ga1):#lambada1后验密度函数
    return pow(lambda1,alafa1)*m.exp(-lambda1*ga1)*m.exp(-lambda1*abs(beta))
    #不含规范化常数的lambda1的后验密度
    #phi(lambda1)*g(beta,lambda1)
    
def houyan2(lambda2,beta,alafa2,ga2):
    return pow(lambda2,alafa2)*m.exp(-lambda2*ga2)*m.exp(-lambda2*pow(beta,2))
    #phi(lambda2)*g(beta,lambda2)


def likehood(z,lambda1,lambda2):#计算似然函数phi(beta|lambda1,lambda2),不含规范化常数
    return 1/m.exp(lambda1 * abs(z)+lambda2 * pow(z,2))

def gamma(x,alafa1,ga1):#不含规范化常熟的伽马分布的密度函数
    return (x**(alafa1-1))*m.exp(-ga1*x)



def ga_lambda1(N,alafa1,ga1):#mh抽取lambda1~gamma(alafa1,gama1),对应double—MH中的第一步
    z=[0]*(N+1)
    t=0
    while t<N:
        t+=1
        cur=random.uniform(0,1)
        next1=random.uniform(0,1)
        u=random.uniform(0,1)
        alpha=min(gamma(next1,alafa1,ga1)/gamma(cur,alafa1,ga1),1)
        if u<=alpha:
            z[t]=next1
        else:
            z[t]=cur
    return z  