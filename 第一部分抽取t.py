# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:13:51 2018

@author: admin
"""


import math as m
import numpy as np
#使用numpy.random.wald 抽取逆高斯分布，百度

def ig_t(beta,sigma,lambda1,lambda2):
    ans=[]
    scale=lambda1/(4*lambda2*sigma^2)
    for i in range(len(beta)):
        mean=m.sqrt(lambda1)/(2*lambda2*abs(beta[i]))
        ans[i]=1 + 1/(np.random.wald(mean,scale,size=1))
    return ans


    