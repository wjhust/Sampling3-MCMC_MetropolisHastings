
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:27:52 2018

@author: wangjian
"""
# -*- coding: utf-8 -*-
'''
Created on 2018年10月15日
@author: user
@attention:  Gibbs Sampling利用条件概率产生符合分布的样本，用于估计分布的期望，边缘分布；是一种在无法精确计算情况下，用计算机模拟的方法。
'''

import math as m
import random
random.seed(12)#是否需要设置种子数

######超参数
alafa1=1
alafa2=1
ga1=1
ga2=1

'''
beta是从前一部分的gibbs抽样获得
'''
beta=2



#########需要传递的参数
lambda1=2
lambda2=2
sigma=1



def houyan1(lambda1,beta,alafa1,ga1,sigma):#lambada1后验密度函数,非向量
    return m.pow(lambda1,alafa1)*m.exp(-lambda1*m.pow(sigma,2)*ga1)*m.exp(-lambda1*abs(beta))
    #不含规范化常数的lambda1的后验密度
    #phi(lambda1)*g(beta,lambda1)
    
def houyan2(lambda2,beta,alafa2,ga2,sigma):
    return pow(lambda2,alafa2)*m.exp(-lambda2*m.pow(sigma,2)*ga2)*m.exp(-lambda2*pow(beta,2))
    #phi(lambda2)*g(beta,lambda2)


def likehood(z,lambda1,lambda2):#计算似然函数phi(beta|lambda1,lambda2),不含规范化常数
    return 1/m.exp(lambda1 * abs(z)+lambda2 * pow(z,2))

def lambda1_mh(N,lambda1):#mh抽取辅助变量z,返回的是z的序列；
    z=[0]*(N+1)
    t=0
    while t<N:
        t+=1
        cur=random.uniform(0,1)
        next1=random.uniform(0,1)
        u=random.uniform(0,1)
        alpha=min(likehood(next1,lambda1,lambda2)/likehood(cur,lambda1,lambda2),1)
        if u<=alpha:
            z[t]=next1
        else:
            z[t]=cur
    return z  

def lambda_1(N_hops,lambda2,lambda1):#抽出来的是lambda1
    states = []
    cur = random.uniform(0,1) # 初始化状态
    for i in range(N_hops):
        states.append(cur)
        next1 = random.uniform(0,1) 
        #从原来的迁移矩阵P采样，这里假设P是一个基于均匀分布的迁移矩阵
        #计算接受率，因为beta分布的常数项会抵消，所以用不带常数项的形式，能大幅提速。而且P是均匀分布，所以也相互抵消了
        
        lists=lambda1_mh(30,lambda1)#辅助变量z
        length=len(lists)
        s1=0
        for i in lists:#这里i就是辅助变量
            s1+=likehood(i,next1,lambda2)/likehood(i,cur,lambda2)
        s1=length/s1#1/R的计算,使用倒数方便下一步拒绝率的计算
        ap=min((houyan1(next1,beta,alafa1,ga1,sigma)/houyan1(cur,beta,alafa1,ga1,sigma))*s1,1)
        u=random.uniform(0,1)
        if u<=ap:
            states.append(next1)
        else:
            states.append(cur)
        
    return states[-1]




def lambda2_mh(N,lambda2):#mh抽取辅助变量z,返回的是z的序列；
    z=[0]*(N+1)
    t=0
    while t<N:
        t+=1
        cur=random.uniform(0,1)
        next1=random.uniform(0,1)
        u=random.uniform(0,1)
        alpha=min(likehood(next1,lambda1,lambda2)/likehood(cur,lambda1,lambda2),1)
        if u<=alpha:
            z[t]=next1
        else:
            z[t]=cur
    return z  



def lambda_2(N_hops,lambda2,lambda1):#抽出来的是lambda2
    states = []
    cur = random.uniform(0,1) # 初始化状态
    for i in range(N_hops):
        states.append(cur)
        next1 = random.uniform(0,1) 
        lists=lambda2_mh(30,lambda2)#辅助变量z
        length=len(lists)
        s1=0
        for i in lists:#这里i就是辅助变量
            s1+=likehood(i,lambda1,next1)/likehood(i,lambda1,cur)
        s1=length/s1#1/R的计算,使用倒数方便下一步拒绝率的计算
        ap=min((houyan2(next1,beta,alafa2,ga2,sigma)/houyan2(cur,beta,alafa2,ga2,sigma))*s1,1)
        #这里计算
        u=random.uniform(0,1)
        if u<=ap:
            states.append(next1)
        else:
            states.append(cur)
    return states[-1]#取最后的一部分状态，保证已经收敛


def gibbs(N_hops):
    res1=[]
    res2=[]
    N=10
    res1.append(lambda_1(N,1,1))
    res2.append(lambda_2(N,1,1))
    for i in range(N_hops):
        res1.append(lambda_1(N,res2[-1],res1[-1]))
        res2.append(lambda_2(N,res1[-1],res2[-1]))
    print("参数lambda1的估计为：",res1)
    print("参数lambda2的估计为：",res2)

print(gibbs(10))
    
    


    




'''
res1=[]
res2=[]
N=10
res1.append(lambda_1(N,1,1))
res2.append(lambda_2(N,1,1))
print(res1)
print(res2)



for i in range(2):
    res1.append(lambda_1(N,res2[-1],res1[-1]))
    res2.append(lambda_2(N,res1[-1],res2[-1]))
print("参数lambda1的估计为：",res1)
print( "参数lambda2的估计为：",res2)
'''



