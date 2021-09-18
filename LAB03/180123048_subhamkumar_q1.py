# -*- coding: utf-8 -*-
"""180123048_subhamkumar_q1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZSeV33fyX6CjbaP1IF--YIOuUW2tmCfh
"""

import numpy as np
import matplotlib.pyplot as plt


S_0 = 100
T = 1
r = 0.08
sigma = 0.2
M_list = [5, 10, 15,20]
option_price = []



def lookback_option(M, u, d, p):
    P = [[[S_0, S_0]]]
    for i in range(M):
        Q = []
        for j in range(len(P[i])):
            q = P[i][j][0]*u*p
            q_max = p*max(P[i][j][1], q/p)
            Q.append([q, q_max])
            q = P[i][j][0]*d*(1-p)
            q_max = (1-p)*max(P[i][j][1], q/(1-p))
            Q.append([q, q_max])
        P.append(Q)
    sol = 0
    for p1 in P[len(P)-1]:
        sol += (p1[1]-p1[0])
    return (sol)*np.exp(-r*T)



for M in M_list:
    dt = T/M
    u = np.exp(sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    d = np.exp(-sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    p = (np.exp(r*dt)-d)/(u-d)
    
    val = lookback_option(M, u, d, p)
    option_price.append(val)
    print('The initial  price of the option for M =',M,'is', val)

#option price vs M graph

M_list = list(range(21))
option_price = [0]

for M in M_list[1:]:
    dt = T/M
    u = np.exp(sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    d = np.exp(-sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    p = (np.exp(r*dt)-d)/(u-d)
    val = lookback_option(M, u, d, p)
    option_price.append(val)

plt.plot(M_list, option_price,color="r")
plt.xlabel('Value of M')
plt.ylabel('Initial  Option Price')
plt.title('Initial  Option Price vs M')
plt.show()

#Option price at different time for M = 5
M = 5
dt = T/M
u = np.exp(sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
d = np.exp(-sigma*np.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
p = (np.exp(r*dt)-d)/(u-d)

mat1 = [[[S_0, S_0]]]
for i in range(M):
    Q = []
    for j in range(len(mat1[i])):
        q = mat1[i][j][0]*u
        q_max = max(mat1[i][j][1], q)
        Q.append([q, q_max])
        q = mat1[i][j][0]*d
        q_max = max(mat1[i][j][1], q)
        Q.append([q, q_max])
    mat1.append(Q)


price_list = []
for P in mat1[len(mat1)-1]:
    price_list.append(P[1]-P[0])

for i in range(6):
    print('At time t =', (5-i)*dt, 'the values of the option price are:\n')
    print(price_list)
    print('\n')
    temp = []
    for i in range(int(len(price_list)/2)):
        temp.append((p*price_list[2*i]+(1-p)*price_list[2*i+1])*np.exp(-r*dt))
    price_list = temp