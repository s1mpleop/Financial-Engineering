# -*- coding: utf-8 -*-
"""Lab10_180123048_q1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1D-Xe0bWbJ5LwrYwttpzka0ENp_9ZXxl9
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,12)

def GBM(init, mu, sigma, T, dt, str):
  if str == 'Normal':
    no_values = round(T/dt)
    values = np.zeros([1,no_values])
    values[0][0] = init
    for i in range(1,no_values):
      c = (mu-0.5*sigma**2)
      values[0][i] = values[0][i-1]*np.exp(c*dt + sigma*np.sqrt(dt)*np.random.normal())
  elif str == 'var_reduce':
    no_values = round(T/dt)
    values1 = np.zeros([1,no_values])
    values2 = np.zeros([1,no_values])
    values1[0][0] = init
    values2[0][0] = init
    for i in range(1,no_values):
      c = (mu-0.5*sigma**2)
      values1[0][i] = values1[0][i-1]*np.exp(c*dt + sigma*np.sqrt(dt)*np.random.normal())   
      values2[0][i] = values2[0][i-1]*np.exp(c*dt + sigma*np.sqrt(dt)*np.random.normal())
    values = (values1 + values2)/2
  return values

init = 100
r = 0.05
mu = 0.1
sigma = 0.2
T = 0.5
dt = 0.001
paths = 10
K = 105
values = np.zeros([round(T/dt),paths])
str = 'Normal'
for i in range(0,paths):
  values[:,i] = GBM(init, mu, sigma, T, dt, str)

print('Check for correctness')
print('Mean',np.mean(values[-1,:]))
print('Variance',np.var(values[-1,:]))

x = np.linspace(0,T-dt,500)
plt.plot(x,values)
plt.xlabel('t')
plt.ylabel('Price')
plt.title('In real world')
plt.show()

for i in range(0,paths):
  values[:,i] = GBM(init, r, sigma, T, dt, str)

print('Check for correctness')
print('Mean',np.mean(values[-1,:]))
print('Variance',np.var(values[-1,:]))

x = np.linspace(0,T-dt,500)
plt.plot(x,values)
plt.xlabel('t')
plt.ylabel('Price')
plt.title('In risk-free world')
plt.show()

K = []
K.append(90)
K.append(105)
K.append(110)
paths = 100

for i in range(0,3):
  call_price = 0
  put_price = 0
  for k in range(0,paths):
    S = GBM(init, r, sigma, T, dt, str) 
    X = (sum(S[0])/(T/dt))-K[i]
    X1 = K[i]-(sum(S[0])/(T/dt))
    call_price = call_price + max(X,0)
    put_price = put_price + max(X1,0)
  cp = (call_price/paths)*np.exp(-r*T)
  pp = (put_price/paths)*np.exp(-r*T)
  print("Price of call option with strike price: %d is %f"%(K[i],cp))
  print("Price of put option with strike price : %d is %f"%(K[i],pp))

init = 100
r = 0.05
mu = 0.1
sigma = 0.2
T = 0.5
dt = 0.001
paths = 10
K = 105

var = np.linspace(80,120,81)
paths = 100
cp = np.zeros([1,81])
pp = np.zeros([1,81])

for i in range(0,81):
  call_price = 0
  put_price = 0
  for k in range(0,paths):
    S = GBM(var[i], r, sigma, T, dt, str)
    Mc = (sum(S[0])/(T/dt))-K
    Mp = K-(sum(S[0])/(T/dt))
    call_price = call_price + max(Mc,0)
    put_price = put_price + max(Mp,0)
  cp[0][i] = (call_price/paths)*np.exp(-r*T)
  pp[0][i] = (put_price/paths)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,cp[0])
plt.xlabel('Initial Stock price')
plt.ylabel('Call Price')

plt.subplot(2,1,2)
plt.plot(var,pp[0])
plt.xlabel('Initial Stock price')
plt.ylabel('Put Price')
plt.show()

paths = 100
cp = np.zeros([1,81])
pp = np.zeros([1,81])

for i in range(0,81):
  call_price = 0
  put_price = 0
  for k in range(0,paths):
    S = GBM(init, r, sigma, T, dt, str)
    Mc = (sum(S[0])/(T/dt))-var[i]
    Mp = var[i]-(sum(S[0])/(T/dt))
    call_price = call_price + max(Mc,0)
    put_price = put_price + max(Mp,0)
  cp[0][i] = (call_price/paths)*np.exp(-r*T)
  pp[0][i] = (put_price/paths)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,cp[0])
plt.xlabel('Strike price')
plt.ylabel('Call Price')

plt.subplot(2,1,2)
plt.plot(var,pp[0])
plt.xlabel('Strike price')
plt.ylabel('Put Price')
plt.show()

var = np.linspace(0.01,0.9,90)
paths = 500
cp = np.zeros([1,90])
pp = np.zeros([1,90])

for i in range(0,90):
  call_price = 0
  put_price = 0
  for k in range(0,paths):
    S = GBM(init, var[i], sigma, T, dt, str)
    Mc = (sum(S[0])/(T/dt))-K
    Mp = K-(sum(S[0])/(T/dt))
    call_price = call_price + max(Mc,0)
    put_price = put_price + max(Mp,0)
  cp[0][i] = (call_price/paths)*np.exp(-r*T)
  pp[0][i] = (put_price/paths)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,cp[0])
plt.xlabel('Risk-free price')
plt.ylabel('Call Price')

plt.subplot(2,1,2)
plt.plot(var,pp[0])
plt.xlabel('Risk-free price')
plt.ylabel('Put Price')
plt.show()

var = np.linspace(0.01,0.9,90)
paths = 500
cp = np.zeros([1,90])
pp = np.zeros([1,90])

for i in range(0,90):
  call_price = 0
  put_price = 0
  for k in range(0,paths):
    S = GBM(init, r, var[i], T, dt, str)
    Mc = (sum(S[0])/(T/dt))-K
    Mp = K-(sum(S[0])/(T/dt))
    call_price = call_price + max(Mc,0)
    put_price = put_price + max(Mp,0)
  cp[0][i] = (call_price/paths)*np.exp(-r*T)
  pp[0][i] = (put_price/paths)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,cp[0])
plt.xlabel('Volatility')
plt.ylabel('Call Price')

plt.subplot(2,1,2)
plt.plot(var,pp[0])
plt.xlabel('Volatility')
plt.ylabel('Put Price')
plt.show()

var = np.linspace(0.01,0.9,18)
paths = 500
cp = np.zeros([1,18])
pp = np.zeros([1,18])

for i in range(0,18):
  call_price = 0
  put_price = 0
  for k in range(0,paths):
    S = GBM(init, r, sigma, var[i], dt, str)
    Mc = (sum(S[0])/(T/dt))-K
    Mp = K-(sum(S[0])/(T/dt))
    call_price = call_price + max(Mc,0)
    put_price = put_price + max(Mp,0)
  cp[0][i] = (call_price/paths)*np.exp(-r*T)
  pp[0][i] = (put_price/paths)*np.exp(-r*T)

plt.subplot(2,1,1)
plt.plot(var,cp[0])
plt.xlabel('Maturity')
plt.ylabel('Call Price')

plt.subplot(2,1,2)
plt.plot(var,pp[0])
plt.xlabel('Maturity')
plt.ylabel('Put Price')
plt.show()