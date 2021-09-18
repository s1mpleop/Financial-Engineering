# -*- coding: utf-8 -*-
"""Lab07_180123048_subhamkumar_q1-q3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1swsj1G_QUdvtt66LVWiRdZii83KZF2Gs
"""

import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def european_option_prices(T,K,S,r,sig,t):
  
  if T == t:
    putp = max(K-S,0)
    callp = max(S-K,0)
    return(callp,putp)
  d_1 = (math.log(S/K)+(r+sig**2/2)*(T-t))/(sig*math.sqrt(T-t))
  d_2 = d_1-sig*math.sqrt(T-t)
  putp = K*math.exp(-r*(T-t))*norm.cdf(-1*d_2) - S*norm.cdf(-1*d_1)
  callp = S*norm.cdf(d_1) - K*norm.cdf(d_2)*math.exp(-r*(T-t))
  return(callp,putp)

def Graph3(X, Y, Z1, XLabel, YLabel, G1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.array(X), np.array(Y), np.array(Z1),cmap=cm.viridis)
    ax.set_xlabel(XLabel, size=10)
    ax.set_ylabel(YLabel, size=10)
    plt.title(G1, size= 15)
    plt.figure(figsize=(16,9))
    plt.show()

T=1
K=1
r=0.05
sig=0.6

t = [0, 0.2, 0.4, 0.6, 0.8,1]
s = np.linspace(0.1,1.5,29)

call_p = np.zeros([len(t),len(s)])
put_p = np.zeros([len(t),len(s)])

for i in range(0,len(t)):
  for j in range(0,len(s)):
    call_p[i][j],put_p[i][j] = european_option_prices(1,1,s[j],0.05,0.6,t[i])

for i in range(0,len(t)):
  plt.plot(s,call_p[i][:],label = str(t[i]))
plt.grid()
plt.xlabel('Stock price')  
plt.ylabel('Option price')
plt.title('Call prices', size= 15)
plt.legend(loc="upper left")
plt.figure(figsize=(16,9))
plt.show()

for i in range(0,len(t)):
  plt.plot(s,put_p[i][:],label = str(t[i]))
plt.grid()
plt.xlabel('Stock price')  
plt.ylabel('Option price')
plt.title('Put prices', size= 15)
plt.legend(loc="upper left")
plt.figure(figsize=(20 ,15))
plt.show()

time = []

for j in range(0,6):
  for i in range(0,29):
    
    time.append(t[j])
  
X  = np.reshape(time, (6, 29))

Y = []
for j in range(0,6):
  Y.append(s)

Y = np.reshape(Y, (6, 29))

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, call_p)
ax.set_xlabel('Time', size=10)
ax.set_ylabel('Stock price', size=10)
plt.title("Call prices", size=15)
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X, Y, put_p)
ax.set_xlabel('Time', size=10)
ax.set_ylabel('Stock price', size=10)
plt.title("Put prices", size=15)
plt.show()

Graph3(X, Y,call_p , "Time", "Stock price", "Call prices")

Graph3(X, Y,put_p , "Time", "Stock price", "Put prices")