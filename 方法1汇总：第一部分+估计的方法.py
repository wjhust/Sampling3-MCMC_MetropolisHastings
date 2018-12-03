# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:40:32 2018

@author: admin
"""

import random
import math as m
import numpy as np

random.seed(12)
'''
============================================================:::::::::::::::第一部分：抽取beta,t,sigma^2::::::::::::::===================================
'''
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
ga1=1
ga2=1
alafa1=2
alafa2=2

#sigma=1

#这里x是2X5矩阵，y是2X1矩阵，beta需要是5X1矩阵


#=========================================多维高斯分布的抽样====================
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

#=================================================逆高斯分布GIG=================
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

#==================================================拒绝接受抽样抽取sigma^2======
    
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
#==============================gibbs循环采样===================================
'''gibbs抽取第一部分参数
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
'''





'''=======================================================================:::第二部分:估计的方法::========================================================================================'''
def houyan1(lambda1,beta,alafa1,ga1,sigma):#lambada1后验密度函数,beta是向量
    return m.pow(lambda1,alafa1)*m.exp(-lambda1*m.pow(sigma,2)*ga1)*m.exp(-lambda1*np.linalg.norm(beta,ord=1))
    #不含规范化常数的lambda1的后验密度
    #phi(lambda1)*g(beta,lambda1)
    
def houyan2(lambda2,beta,alafa2,ga2,sigma):#beta是向量
    return m.pow(lambda2,alafa2)*m.exp(-lambda2*m.pow(sigma,2)*ga2)*m.exp(-lambda2*np.linalg.norm(beta,ord=2))
    #phi(lambda2)*g(beta,lambda2)


def likehood(z,lambda1,lambda2):#计算似然函数phi(beta|lambda1,lambda2),不含规范化常数
    return 1/m.exp(lambda1 * abs(z)+lambda2 * pow(z,2))

def lambda1_mh(N,lambda1,lambda2):#mh抽取辅助变量z,返回的是z的序列；
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

def lambda_1(N_hops,beta,alafa1,ga1,sigma,lambda2,lambda1):#抽出来的是lambda1
    states = []
    cur = random.uniform(0,1) # 初始化状态
    for i in range(N_hops):
        states.append(cur)
        next1 = random.uniform(0,1) 
        #从原来的迁移矩阵P采样，这里假设P是一个基于均匀分布的迁移矩阵
        #计算接受率，因为beta分布的常数项会抵消，所以用不带常数项的形式，能大幅提速。而且P是均匀分布，所以也相互抵消了
        
        lists=lambda1_mh(30,lambda1,lambda2)#辅助变量z
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




def lambda2_mh(N,lambda2,lambda1):#mh抽取辅助变量z,返回的是z的序列；
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



def lambda_2(N_hops,lambda2,lambda1,beta,alafa2,ga2,sigma):#抽出来的是lambda2
    states = []
    cur = random.uniform(0,1) # 初始化状态
    for i in range(N_hops):
        states.append(cur)
        next1 = random.uniform(0,1) 
        lists=lambda2_mh(30,lambda2,lambda1)#辅助变量z
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

'''#gibbs抽取参数lambda
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





def gibbs_1(N_hops):
    lambda1=[0.1]
    lambda2=[0.1]
    beta_ans=[]
    sigma_ans=[0.1]
    t_ans=[[2,3,4,5,6]]
    for i in range(N_hops):
        beta_ans.append(gauss_beta(y,x,t_ans[-1],lambda2[-1],sigma_ans[-1],p))
        sigma_ans.append(ac_re_sigma(y,x,t_ans[-1],beta_ans[-1],lambda1[-1],lambda2[-1],n,p))
        t_ans.append(ig_t(beta_ans[-1],sigma_ans[-1],lambda1[-1],lambda2[-1]))
    #这里beta的格式为矩阵形式，无法进行下一步的计算,并且是5X1矩阵,需要变成list
    b3=np.transpose(np.asarray(beta_ans[-1]))
    beta_tran=[b3[0][i] for i in range(len(b3[0]))]
    for i in range(N_hops):
        lambda1.append(lambda_1(N_hops,beta_tran,alafa1,ga1,sigma_ans[-1],lambda2[-1],lambda1[-1]))
        lambda2.append(lambda_2(N_hops,lambda2[-1],lambda1[-1],beta_tran,alafa2,ga2,sigma_ans[-1])) 
    print("参数beta的估计为：",beta_tran)
    print("参数sigma的估计为:",sigma_ans[-1])
    print("参数t的估计:",t_ans[-1])    
    print("参数lambda1的估计为：",lambda1[-1])
    print("参数lambda2的估计为：",lambda2[-1])
    
gibbs_1(50) 
    