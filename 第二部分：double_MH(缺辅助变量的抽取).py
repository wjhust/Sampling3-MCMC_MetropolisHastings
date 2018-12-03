# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:45:01 2018

@author: admin
"""
import random
import math as m
alafa1=1
ga1=1
n=len(y)#y的维度

def gamma(x,alafa1,ga1):#不含规范化常熟的伽马分布的密度函数
    return (x**(alafa1-1))*m.exp(-ga1*x)


def lambda1_phi(x,lambda1):
    return m.exp(-lambda1*abs(x))


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








def fuzhu_y(beta,n,lambda1):#抽取辅助变量,初始是beta，在lambda1情况下beta经过n步MH更新为辅助变量z,即beta_m=y
    for i in range(n):
        z=[0]*(n)
        cur=random.uniform(0,1)
        next1=random.uniform(0,1)
        u=random.uniform(0,1)
        alpha=min(m.exp(-lambda1 * abs(next1))/m.exp(-lambda1 * abs(cur)),1)
        if u<alpha:
            z[i]=next1
        else:
            z[i]=cur
    return z[-1]

def double_mh_lambda1(N,beta):
    #beta是自变量,z是辅助变量
    states=[]
    cur=random.uniform(0,1)#当前的lambda1
    next1=ga_lambda1(N,alafa1,ga1)[-1]#第一步抽出lambda1*
    states.append(next1)
    #这里第一个抽出的lambda1不管接受与否均加入到states中，因为我们只需要states最后的几个即可，前边无所谓
    for i in range(N):
        z=fuzhu_y(beta,n,states[-1])#抽取辅助变量z
        ap=min((lambda1_phi(z,cur)/lambda1_phi(beta,cur))*(lambda1_phi(beta,next1)/lambda1_phi(z,next1)),1)
        u=random.uniform(0,1)
        if u<=ap:
            states.append(next1)
        else:
            states.append(cur)
    return states[-1]
        
        
        
    
    