#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


S_0 = 100
T = 1
r = 0.08
sigma = 0.2
M_list = [5, 10, 25, 50]
Map = {}

def efficient(u, d, p, M, n, S, mx):
    if (S, mx) in Map:
        return Map[(S, mx)]
    
    if n == M:
        Map[(S, mx)] = mx-S
        return mx-S
    
    U = efficient(u, d, p, M, n+1, S*u, max(mx, S*u))
    D = efficient(u, d, p, M, n+1, S*d, max(mx, S*d))
    fin = (p*U + (1-p)*D)*np.exp(-r*T/M)
    Map[(S, mx)] = fin
    return fin

for M in M_list:
    dt = T/M
    u = np.exp(sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    d = np.exp(-sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    p = (np.exp(r*dt)-d)/(u-d)
    Map.clear()
    val = efficient(u, d, p, M, 0, S_0, S_0)
    print('For M =', M, 'the initial value of the lookback option is', val)


# In[ ]:




