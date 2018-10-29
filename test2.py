# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:45:01 2018

@author: admin
"""
import random
import math as m
alafa1=1
ga1=1

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

print(ga_lambda1(10,1,1))