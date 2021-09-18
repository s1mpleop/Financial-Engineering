#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


S_0 = 100
K = 100
T = 1
M_list = [5, 10, 25, 50]
r = 0.08
sigma = 0.2

def efficient(S_0, K, M, r, sigma, u, d, p):
    call_list = [0]*(M+1)
    for i in range(M+1):
        call_list[i] = max(S_0*(u**i)*(d**(M-i)) - K, 0)
    for i in range(M):
        for j in range(M-i):
            call_list[j] = ((1-p)*call_list[j] + p*call_list[j+1])*np.exp(-r*T/M)
    call = call_list[0]
    return call

def normal(S_0, K, M, r, sigma, u, d, p):
    P = [[[S_0, K]]]
    for i in range(M):
        Q = []
        for el in P[i]:
            Q.append([el[0]*u*p, el[1]*p])
            Q.append([el[0]*d*(1-p), el[1]*(1-p)])
        P.append(Q)
    sol = 0
    for el in P[len(P)-1]:
        sol += max(el[0]-el[1], 0)
    return sol*np.exp(-r*T/M)

for M in M_list:
    dt = T/M
    u = np.exp(sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    d = np.exp(-sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    p = (np.exp(r*dt)-d)/(u-d)
    if M < 25:
        val = normal(S_0, K, M, r, sigma, u, d, p)
        print('For M =',M,'the value of European Call using normal method is\n', val)
        print('\n')
    else:
        print('For M =',M,"Normal method cannot handle\n")
        print('\n')
    val = efficient(S_0, K, M, r, sigma, u, d, p)
    print('For M =',M,'the value of European Call using efficient method is\n', val)
    print('\n')


# In[ ]:





# In[ ]:




