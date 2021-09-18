#!/usr/bin/env python
# coding: utf-8

# In[10]:


import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def nCr(n, r): 
    return (math.factorial(n)/(math.factorial(r)*math.factorial(n-r))) 
   

def solve(t,S_at_0,K,T,M,r,Sigma):
    Delta_t= (T*1.0)/(M*1.0)

    
    if t==1:
        u= math.exp( (Sigma*math.sqrt(Delta_t)) ) 
        d= math.exp( -1*(Sigma*math.sqrt(Delta_t)) )
    if t==2:
        u= math.exp( (Sigma*math.sqrt(Delta_t)) + ((r-(Sigma*Sigma)/2.0)*Delta_t) )
        d= math.exp( -1*(Sigma*math.sqrt(Delta_t)) + ((r-(Sigma*Sigma)/2.0)*Delta_t) )
    pStar= (math.exp(r*Delta_t)-d)/(u-d)
    qStar= 1-pStar
    Price= []
    P= [S_at_0*1.0]
    Price.append(P)
    for i in range(0, M):
        P= []
        for j in range(0, len(Price[i])):
            P.append(Price[i][j]*u)
            P.append(Price[i][j]*d)
        Price.append(P)
    
    for i in range(0, M):
        for j in range(0, len(Price[i])):
            Price[i+1][2*j]+= Price[i][j]
            Price[i+1][2*j+1]+= Price[i][j]
    Call= []
    Put= []
    for i in range(0, len(Price[len(Price)-1])):
        Call.append(max(0,Price[len(Price)-1][i]/(M+1)-K))
        Put.append(max(0,K-Price[len(Price)-1][i]/(M+1)))
    
    while(len(Call)!=1):
        C= []
        P= []
        for j in range(0, len(Call), 2):
            C.append((pStar*Call[j]+qStar*Call[j+1])/math.exp(r*Delta_t))
            P.append((pStar*Put[j]+qStar*Put[j+1])/math.exp(r*Delta_t))
        Call= C
        Put= P   
    return [Call[0], Put[0]]

def Graph(E_Call, E_Put, XAxis, Xlabel, Ylabel, Glabel):
    plt.plot(XAxis, E_Call, label="European Call Price",color="r")
    plt.plot(XAxis, E_Put, label="European Put Price",color="g")
    plt.xlabel(Xlabel, size=10)
    plt.ylabel(Ylabel, size=10)
    plt.title(Glabel,size= 10)
    plt.legend()
    plt.show()

def Graph2(E_Call_95, E_Call_100, E_Call_105, E_Put_95, E_Put_100, E_Put_105, XAxis):
    plt.plot(M_G, E_Call_95, label="European Call Price (k=95)",color="b")
    plt.plot(M_G, E_Put_95, label="European Put Price (k=95)",color="g")
    plt.plot(M_G, E_Call_100, label="European Call Price (k=100)",color="r")
    plt.plot(M_G, E_Put_100, label="European Put Price (k=100)",color="c")
    plt.plot(M_G, E_Call_105, label="European Call Price (k=105)",color="m")
    plt.plot(M_G, E_Put_105, label="European Put Price (k=105)",color="y")
    plt.xlabel('M', size=10)
    plt.ylabel('Price', size=10)
    plt.title("Option price varying( M (for 3 diff. values of k) )",size= 10)
    plt.legend()
    plt.show()

def Graph3(X, Y, Z1, Z2, XLabel, YLabel, G1, G2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.array(X), np.array(Y), np.array(Z1),cmap=cm.coolwarm)
    ax.set_xlabel(XLabel, size=10)
    ax.set_ylabel(YLabel, size=10)
    ax.set_zlabel('Price', size=10)
    plt.title(G1, size= 10)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.array(X), np.array(Y), np.array(Z2),cmap=cm.coolwarm)
    ax.set_xlabel(XLabel, size=10)
    ax.set_ylabel(YLabel, size=10)
    ax.set_zlabel('Price', size=10)
    plt.title(G2, size= 10)
    plt.show()

    
S_at_0= 100
K= 100
T= 1
r= 8/100
Sigma= 20/100
M= 10

Price= solve(1,S_at_0,K,T,M,r,Sigma)
print("Path dependent derivative chosen: Asian Option")
print("Value of Call Option for set 1: ",(Price[0]))
print("Value of Putt Option for set 1: ",(Price[1]))
print("")

E_Call= []
E_Put= []
S_at_0_G= []
for S_at_0_Graph in range(0, 200):
    Price= solve(1,S_at_0_Graph,K,T,M,r,Sigma)
    E_Call.append(Price[0])
    E_Put.append(Price[1])
    S_at_0_G.append(S_at_0_Graph)
Graph(E_Call, E_Put, S_at_0_G,"S(0)", "Price", "Option price varying( S(0) )")

E_Call= []
E_Put= []
K_G= []
for K_Graph in range(0, 200):
    Price= solve(1,S_at_0,K_Graph,T,M,r,Sigma)
    E_Call.append(Price[0])
    E_Put.append(Price[1])
    K_G.append(K_Graph)
Graph(E_Call, E_Put, K_G,"K", "Price", "Option price varying( K )")

E_Call= []
E_Put= []
r_G= []
for r_Graph in range(100, 1000, 5):
    Price= solve(1,S_at_0,K,T,M,r_Graph/10000,Sigma)
    E_Call.append(Price[0])
    E_Put.append(Price[1])
    r_G.append(r_Graph/10000)
Graph(E_Call, E_Put, r_G,"r", "Price", "Option price varying( r )")

E_Call= []
E_Put= []
Sigma_G= []
for Sigma_Graph in range(1000, 2000, 5):
    Price= solve(1,S_at_0,K,T,M,r,Sigma_Graph/10000)
    E_Call.append(Price[0])
    E_Put.append(Price[1])
    Sigma_G.append(Sigma_Graph/10000)
Graph(E_Call, E_Put, Sigma_G,"Sigma", "Price", "Option price varying( Sigma )")

E_Call_95= []
E_Put_95= []
E_Call_100= []
E_Put_100= []
E_Call_105= []
E_Put_105= []
M_G= []
for M_Graph in range(1, 20):
    Price= solve(1,S_at_0,95,T,M_Graph,r,Sigma)
    E_Call_95.append(Price[0])
    E_Put_95.append(Price[1])
    
    Price= solve(1,S_at_0,100,T,M_Graph,r,Sigma)
    E_Call_100.append(Price[0])
    E_Put_100.append(Price[1])
    
    Price= solve(1,S_at_0,105,T,M_Graph,r,Sigma)
    E_Call_105.append(Price[0])
    E_Put_105.append(Price[1])
    
    M_G.append(M_Graph)
Graph2(E_Call_95, E_Call_100, E_Call_105, E_Put_95, E_Put_100, E_Put_105, M_G)


S_at_0_G= []
K_G= []
E_Call= []
E_Put= []
for S_at_0_Graph in range(50, 100):
    a= []
    b= []
    c= []
    d= []
    for K_Graph in range(50, 100):
        a.append(S_at_0_Graph)
        b.append(K_Graph)
        Price= solve(1,S_at_0_Graph,K_Graph,T,M,r,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    S_at_0_G.append(a)
    K_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(S_at_0_G, K_G, E_Call, E_Put, "S", "K", "Call Price varying( S,k )", "Put Price varying( S,k )")

S_at_0_G= []
M_G= []
E_Call= []
E_Put= []
for S_at_0_Graph in range(70, 90):
    a= []
    b= []
    c= []
    d= []
    for M_Graph in range(1, 10):
        a.append(S_at_0_Graph)
        b.append(M_Graph)
        Price= solve(1,S_at_0_Graph,K,T,M_Graph,r,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    S_at_0_G.append(a)
    M_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(S_at_0_G, M_G, E_Call, E_Put, "S", "M", "Call Price varying( S,M )", "Put Price varying( S,M )")
    
K_G= []
M_G= []
E_Call= []
E_Put= []
for K_Graph in range(70, 90):
    a= []
    b= []
    c= []
    d= []
    for M_Graph in range(1, 10):
        a.append(K_Graph)
        b.append(M_Graph)
        Price= solve(1,S_at_0,K_Graph,T,M_Graph,r,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    K_G.append(a)
    M_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(K_G, M_G, E_Call, E_Put, "K", "M", "Call Price varying( K,M )", "Put Price varying( K,M )")


###################################################################################
#####################
##################
#############################################################################################
#####################
##################
##########


r_G= []
Sigma_G= []
E_Call= []
E_Put= []
for r_Graph in range(100, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for Sigma_Graph in range(1000, 2000, 10):
        a.append(r_Graph/10000)
        b.append(Sigma_Graph/10000)
        Price= solve(1,S_at_0,K,T,M,r_Graph/10000,Sigma_Graph/10000)
        c.append(Price[0])
        d.append(Price[1])
    r_G.append(a)
    Sigma_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(r_G, Sigma_G, E_Call, E_Put, "r", "Sigma", "Call Price varying( r,Sigma )", "Put Price varying( r,Sigma )")

r_G= []
S_at_0_G= []
E_Call= []
E_Put= []
for r_Graph in range(100, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for S_at_0_Graph in range(70, 90):
        a.append(r_Graph/10000)
        b.append(S_at_0_Graph)
        Price= solve(1,S_at_0_Graph,K,T,M,r_Graph/10000,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    r_G.append(a)
    S_at_0_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(r_G, S_at_0_G, E_Call, E_Put, "r", "S(0)", "Call Price varying( r,S(0) )", "Put Price varying( r,S(0) )")

r_G= []
K_G= []
E_Call= []
E_Put= []
for r_Graph in range(100, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for K_Graph in range(70, 90):
        a.append(r_Graph/10000)
        b.append(K_Graph)
        Price= solve(1,S_at_0,K_Graph,T,M,r_Graph/10000,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    r_G.append(a)
    K_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(r_G, K_G, E_Call, E_Put, "r", "K", "Call Price varying( r,K )", "Put Price varying( r,K )")

r_G= []
M_G= []
E_Call= []
E_Put= []
for r_Graph in range(100, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for M_Graph in range(1, 10):
        a.append(r_Graph/10000)
        b.append(M_Graph)
        Price= solve(1,S_at_0,K,T,M_Graph,r_Graph/10000,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    r_G.append(a)
    M_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(r_G, M_G, E_Call, E_Put, "r", "M", "Call Price varying( r,M )", "Put Price varying( r,M )")

Sigma_G= []
S_at_0_G= []
E_Call= []
E_Put= []
for Sigma_Graph in range(1000, 2000, 10):
    a= []
    b= []
    c= []
    d= []
    for S_at_0_Graph in range(70, 90):
        a.append(Sigma_Graph/10000)
        b.append(S_at_0_Graph)
        Price= solve(1,S_at_0_Graph,K,T,M,r,Sigma_Graph/10000)
        c.append(Price[0])
        d.append(Price[1])
    Sigma_G.append(a)
    S_at_0_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(Sigma_G, S_at_0_G, E_Call, E_Put, "Sigma", "S(0)", "Call Price varying( Sigma,S(0) )", "Put Price varying( Sigma,S(0) )")

Sigma_G= []
K_G= []
E_Call= []
E_Put= []
for Sigma_Graph in range(1000, 2000, 10):
    a= []
    b= []
    c= []
    d= []
    for K_Graph in range(70, 90):
        a.append(Sigma_Graph/10000)
        b.append(K_Graph)
        Price= solve(1,S_at_0,K_Graph,T,M,r,Sigma_Graph/10000)
        c.append(Price[0])
        d.append(Price[1])
    Sigma_G.append(a)
    K_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(Sigma_G, K_G, E_Call, E_Put, "Sigma", "K", "Call Price varying( Sigma,K )", "Put Price varying( Sigma,K )")

Sigma_G= []
M_G= []
E_Call= []
E_Put= []
for Sigma_Graph in range(1000, 2000, 10):
    a= []
    b= []
    c= []
    d= []
    for M_Graph in range(1, 10):
        a.append(Sigma_Graph/10000)
        b.append(M_Graph)
        Price= solve(1,S_at_0,K,T,M_Graph,r,Sigma_Graph/10000)
        c.append(Price[0])
        d.append(Price[1])
    Sigma_G.append(a)
    M_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(Sigma_G, M_G, E_Call, E_Put, "Sigma", "M", "Call Price varying( Sigma,M )", "Put Price varying( Sigma,M )")

################################################################################

Price=solve(2,S_at_0,K,T,M,r,Sigma)
print("Value of Call Option for set 2: ",(Price[0]))
print("Value of Option for set 2: ",(Price[1]))
print("")

E_Call= []
E_Put= []
S_at_0_G= []
for S_at_0_Graph in range(0, 200):
    Price= solve(2,S_at_0_Graph,K,T,M,r,Sigma)
    E_Call.append(Price[0])
    E_Put.append(Price[1])
    S_at_0_G.append(S_at_0_Graph)
Graph(E_Call, E_Put, S_at_0_G,"S(0)", "Price", "Option price varying( S(0) )")

E_Call= []
E_Put= []
K_G= []
for K_Graph in range(0, 200):
    Price= solve(2,S_at_0,K_Graph,T,M,r,Sigma)
    E_Call.append(Price[0])
    E_Put.append(Price[1])
    K_G.append(K_Graph)
Graph(E_Call, E_Put, K_G,"K", "Price", "Option price varying( K )")

E_Call= []
E_Put= []
r_G= []
for r_Graph in range(100, 1000, 5):
    Price= solve(2,S_at_0,K,T,M,r_Graph/10000,Sigma)
    E_Call.append(Price[0])
    E_Put.append(Price[1])
    r_G.append(r_Graph/10000)
Graph(E_Call, E_Put, r_G,"r", "Price", "Option price varying( r )")

E_Call= []
E_Put= []
Sigma_G= []
for Sigma_Graph in range(1000, 2000, 5):
    Price= solve(2,S_at_0,K,T,M,r,Sigma_Graph/10000)
    E_Call.append(Price[0])
    E_Put.append(Price[1])
    Sigma_G.append(Sigma_Graph/10000)
Graph(E_Call, E_Put, Sigma_G,"Sigma", "Price", "Option price varying( Sigma )")

E_Call_95= []
E_Put_95= []
E_Call_100= []
E_Put_100= []
E_Call_105= []
E_Put_105= []
M_G= []
for M_Graph in range(1, 10):
    Price= solve(2,S_at_0,95,T,M_Graph,r,Sigma)
    E_Call_95.append(Price[0])
    E_Put_95.append(Price[1])
    
    Price= solve(2,S_at_0,100,T,M_Graph,r,Sigma)
    E_Call_100.append(Price[0])
    E_Put_100.append(Price[1])
    
    Price= solve(2,S_at_0,105,T,M_Graph,r,Sigma)
    E_Call_105.append(Price[0])
    E_Put_105.append(Price[1])
    
    M_G.append(M_Graph)
Graph2(E_Call_95, E_Call_100, E_Call_105, E_Put_95, E_Put_100, E_Put_105, M_G)

S_at_0_G= []
K_G= []
E_Call= []
E_Put= []
for S_at_0_Graph in range(50, 70):
    a= []
    b= []
    c= []
    d= []
    for K_Graph in range(50, 70):
        a.append(S_at_0_Graph)
        b.append(K_Graph)
        Price= solve(2,S_at_0_Graph,K_Graph,T,M,r,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    S_at_0_G.append(a)
    K_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(S_at_0_G, K_G, E_Call, E_Put, "S", "K", "Call Price varying( S,k )", "Put Price varying( S,k )")

S_at_0_G= []
M_G= []
E_Call= []
E_Put= []
for S_at_0_Graph in range(70, 90):
    a= []
    b= []
    c= []
    d= []
    for M_Graph in range(1, 10):
        a.append(S_at_0_Graph)
        b.append(M_Graph)
        Price= solve(2,S_at_0_Graph,K,T,M_Graph,r,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    S_at_0_G.append(a)
    M_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(S_at_0_G, M_G, E_Call, E_Put, "S", "M", "Call Price varying( S,M )", "Put Price varying( S,M )")
    
K_G= []
M_G= []
E_Call= []
E_Put= []
for K_Graph in range(70, 90):
    a= []
    b= []
    c= []
    d= []
    for M_Graph in range(1, 10):
        a.append(K_Graph)
        b.append(M_Graph)
        Price= solve(2,S_at_0,K_Graph,T,M_Graph,r,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    K_G.append(a)
    M_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(K_G, M_G, E_Call, E_Put, "K", "M", "Call Price varying( K,M )", "Put Price varying( K,M )")

r_G= []
Sigma_G= []
E_Call= []
E_Put= []
for r_Graph in range(100, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for Sigma_Graph in range(1000, 2000, 10):
        a.append(r_Graph/10000)
        b.append(Sigma_Graph/10000)
        Price= solve(2,S_at_0,K,T,M,r_Graph/10000,Sigma_Graph/10000)
        c.append(Price[0])
        d.append(Price[1])
    r_G.append(a)
    Sigma_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(r_G, Sigma_G, E_Call, E_Put, "r", "Sigma", "Call Price varying( r,Sigma )", "Put Price varying( r,Sigma )")

r_G= []
S_at_0_G= []
E_Call= []
E_Put= []
for r_Graph in range(100, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for S_at_0_Graph in range(70, 90):
        a.append(r_Graph/10000)
        b.append(S_at_0_Graph)
        Price= solve(2,S_at_0_Graph,K,T,M,r_Graph/10000,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    r_G.append(a)
    S_at_0_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(r_G, S_at_0_G, E_Call, E_Put, "r", "S(0)", "Call Price varying( r,S(0) )", "Put Price varying( r,S(0) )")

r_G= []
K_G= []
E_Call= []
E_Put= []
for r_Graph in range(100, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for K_Graph in range(70, 90):
        a.append(r_Graph/10000)
        b.append(K_Graph)
        Price= solve(2,S_at_0,K_Graph,T,M,r_Graph/10000,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    r_G.append(a)
    K_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(r_G, K_G, E_Call, E_Put, "r", "K", "Call Price varying( r,K )", "Put Price varying( r,K )")

r_G= []
M_G= []
E_Call= []
E_Put= []
for r_Graph in range(100, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for M_Graph in range(1, 10):
        a.append(r_Graph/10000)
        b.append(M_Graph)
        Price= solve(2,S_at_0,K,T,M_Graph,r_Graph/10000,Sigma)
        c.append(Price[0])
        d.append(Price[1])
    r_G.append(a)
    M_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(r_G, M_G, E_Call, E_Put, "r", "M", "Call Price varying( r,M )", "Put Price varying( r,M )")

Sigma_G= []
S_at_0_G= []
E_Call= []
E_Put= []
for Sigma_Graph in range(1000, 2000, 10):
    a= []
    b= []
    c= []
    d= []
    for S_at_0_Graph in range(70, 100):
        a.append(Sigma_Graph/10000)
        b.append(S_at_0_Graph)
        Price= solve(2,S_at_0_Graph,K,T,M,r,Sigma_Graph/10000)
        c.append(Price[0])
        d.append(Price[1])
    Sigma_G.append(a)
    S_at_0_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(Sigma_G, S_at_0_G, E_Call, E_Put, "Sigma", "S(0)", "Call Price varying( Sigma,S(0) )", "Put Price varying( Sigma,S(0) )")

Sigma_G= []
K_G= []
E_Call= []
E_Put= []
for Sigma_Graph in range(1000, 2000, 10):
    a= []
    b= []
    c= []
    d= []
    for K_Graph in range(70, 90):
        a.append(Sigma_Graph/10000)
        b.append(K_Graph)
        Price= solve(2,S_at_0,K_Graph,T,M,r,Sigma_Graph/10000)
        c.append(Price[0])
        d.append(Price[1])
    Sigma_G.append(a)
    K_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(Sigma_G, K_G, E_Call, E_Put, "Sigma", "K", "Call Price varying( Sigma,K )", "Put Price varying( Sigma,K )")

Sigma_G= []
M_G= []
E_Call= []
E_Put= []
for Sigma_Graph in range(1000, 2000, 10):
    a= []
    b= []
    c= []
    d= []
    for M_Graph in range(1, 10):
        a.append(Sigma_Graph/10000)
        b.append(M_Graph)
        Price= solve(2,S_at_0,K,T,M_Graph,r,Sigma_Graph/10000)
        c.append(Price[0])
        d.append(Price[1])
    Sigma_G.append(a)
    M_G.append(b)
    E_Call.append(c)
    E_Put.append(d)
Graph3(Sigma_G, M_G, E_Call, E_Put, "Sigma", "M", "Call Price varying( Sigma,M )", "Put Price varying( Sigma,M )")


# In[ ]:




