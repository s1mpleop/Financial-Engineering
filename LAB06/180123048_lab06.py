# -*- coding: utf-8 -*-
"""180123048_lab06.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ibCwuJMtt7NXtK8_bis5lT7ai9Vw-JEF
"""

from google.colab import files
files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt
import scipy.stats as stats
from pandas import to_datetime



stock = ['AXBK.csv','HDFC.csv','ICBK.csv','ITC.csv',
            'MRTI.csv','RELI.csv','SBI (1).csv','SUN.csv',
            'TAMO.csv','YESB.csv','ASPN.NS.csv','AXBK.NS.csv',
            'BAJA.NS.csv','BPCL.NS.csv','CIPL.NS.csv','COAL.NS.csv','HLL.NS.csv'
            ,'INFY.NS.csv','PGRD.NS.csv','ZEE.NS.csv']


def dailyPrice(stock_name):
    temp = pd.read_csv(stock_name)
    temp.dropna(subset=['Close'], inplace=True)
    res = np.arange(len(temp))
    val = np.array(temp['Close'])
    return[res,val]


def weeklyPrice(stock_name):
    temp = pd.read_csv(stock_name)
    temp.dropna(subset=['Close'], inplace=True)
    temp['Day'] = (to_datetime(temp['Date'])).dt.day_name()
    answer = temp.loc[temp['Day'] == 'Monday']
    res = np.arange(len(answer))
    val = np.array(answer['Close'])
    return[res,val]



def monthlyPrice(stock_name):
    df = pd.read_csv(stock_name)
    df.dropna(subset=['Close'], inplace=True)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    temp = df.resample('1M').mean()
    res = np.arange(len(temp))
    val = np.array(temp['Close'])
    return[res,val]


def weeklyReturns(stock_name):
    temp = pd.read_csv(stock_name)
    temp.dropna(subset=['Close'], inplace=True)
    temp['Day'] = (to_datetime(temp['Date'])).dt.day_name()
    answer = temp.loc[temp['Day'] == 'Monday']
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = (res1-res2)/res2
    avg = np.average(res)
    var = np.std(res)
    norm_ret = (res - avg)/var
    meu, sigma = 0, 1
    res = np.linspace(min(norm_ret), max(norm_ret), len(norm_ret))
    val = (1/(2*np.pi*(sigma**2))**0.5)*np.exp(-(res-meu)**2/(sigma)**2)
    return[norm_ret,res,val]

def monthlyReturns(stock_name):
    df = pd.read_csv(stock_name)
    df.dropna(subset=['Close'], inplace=True)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    answer = df.resample('1M').mean()
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = (res1-res2)/res2
    avg = np.average(res)
    var = np.std(res)
    norm_ret = (res - avg)/var
    meu, sigma = 0, 1
    res = np.linspace(min(norm_ret), max(norm_ret), len(norm_ret))
    val = (1/(2*np.pi*(sigma**2))**0.5)*np.exp(-(res-meu)**2/(sigma)**2)
    return[norm_ret,res,val]


def dailyReturns(stock_name):
    answer = pd.read_csv(stock_name)
    answer.dropna(subset=['Close'], inplace=True)
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = (res1-res2)/res2
    avg = np.average(res)
    var = np.std(res)
    norm_ret = (res - avg)/var
    meu, sigma = 0, 1
    res = np.linspace(min(norm_ret), max(norm_ret), len(norm_ret))
    val = (1/(2*np.pi*(sigma**2))**0.5)*np.exp(-(res-meu)**2/(sigma)**2)
    plt.plot(res,val)
    return[norm_ret,res,val]


def dailyLogReturns(stock_name):
    answer = pd.read_csv(stock_name)
    answer.dropna(subset=['Close'], inplace=True)
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = (res1-res2)/res2
    logres = np.log(1+res)
    avg = np.average(logres)
    var = np.std(logres)
    norm_ret = (logres - avg)/var
    meu, sigma = 0, 1
    res = np.linspace(min(norm_ret), max(norm_ret), len(norm_ret))
    val = (1/(2*np.pi*(sigma**2))**0.5)*np.exp(-(res-meu)**2/(sigma)**2)
    return[norm_ret,res,val]

def monthlyLogReturns(stock_name):
    df = pd.read_csv(stock_name)
    df.dropna(subset=['Close'], inplace=True)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    answer = df.resample('1M').mean()
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = np.log(res1/res2)
    avg = np.average(res)
    var = np.std(res)
    norm_ret = (res - avg)/var
    meu, sigma = 0, 1
    res = np.linspace(min(norm_ret), max(norm_ret), len(norm_ret))
    val = (1/(2*np.pi*(sigma**2))**0.5)*np.exp(-(res-meu)**2/(sigma)**2)
    return[norm_ret,res,val]


def weeklyLogReturns(stock_name):
    temp = pd.read_csv(stock_name)
    temp.dropna(subset=['Close'], inplace=True)
    temp['Day'] = (to_datetime(temp['Date'])).dt.day_name()
    answer = temp.loc[temp['Day'] == 'Monday']
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = np.log(res1/res2)
    avg = np.average(res)
    var = np.std(res)
    norm_ret = (res - avg)/var
    meu, sigma = 0, 1
    res = np.linspace(min(norm_ret), max(norm_ret), len(norm_ret))
    val = (1/(2*np.pi*(sigma**2))**0.5)*np.exp(-(res-meu)**2/(sigma)**2)
    return[norm_ret,res,val]


def predPriceDaily(stock_name):
    answer = pd.read_csv(stock_name)
    answer.dropna(subset=['Close'], inplace=True)
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = (res1-res2)/res2
    res = np.log(1+res)
    final_res = res[:987]
    meu = np.mean(final_res)
    sigma = np.std(final_res)
    n = len(res)-987
    phhii = np.random.normal(0,1,n)
    W = np.zeros(n)
    W[0] = 0
    for i in range(1, n):
        W[i] = W[i-1]+phhii[i]
    S = np.zeros(n)
    S[0] = answer['Close'][987]
    for i in range(1, n):
        S[i] = S[0]*np.exp(sigma*W[i]+(meu-0.5*(sigma**2))*i/240)
    actual_price = np.array(answer['Close'])
    predicted_price = actual_price[987:]
    Y2 = predicted_price
    return[S,Y2]


def predPricesWeekly(stock_name):
    temp = pd.read_csv(stock_name)
    temp.dropna(subset=['Close'], inplace=True)
    temp['Day'] = (to_datetime(temp['Date'])).dt.day_name()
    answer = temp.loc[temp['Day'] == 'Monday']
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = (res1-res2)/res2
    res = np.log(1+res)
    final_res = res[:208]
    meu = np.mean(final_res)
    sigma = np.std(final_res)
    n = len(res)-208
    phhii = np.random.normal(0,1,n)
    W = np.zeros(n)
    W[0] = 0
    for i in range(1, n):
        W[i] = W[i-1]+phhii[i]
    S = np.zeros(n)
    S[0] = answer['Close'][208]
    for i in range(1, n):
        S[i] = S[0]*np.exp(sigma*W[i]+(meu-0.5*(sigma**2))*i/240)
    actual_price = np.array(answer['Close'])
    predicted_price = actual_price[208:]
    Y2 = predicted_price
    return[S,Y2]

def predPriceMonthly(stock_name):
    df = pd.read_csv(stock_name)
    df.dropna(subset=['Close'], inplace=True)
    df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    answer = df.resample('1M').mean()
    length = len(answer['Close'])
    res1 = np.array(answer['Close'][1:])
    res2 = np.array(answer['Close'][:length-1])
    res = (res1-res2)/res2
    res = np.log(1+res)
    final_res = res[:48]
    meu = np.mean(final_res)
    sigma = np.std(final_res)
    n = len(res)-48
    phhii = np.random.normal(0,1,n)
    W = np.zeros(n)
    W[0] = 0
    for i in range(1, n):
        W[i] = W[i-1]+phhii[i]
    S = np.zeros(n)
    S[0] = answer['Close'][48]
    for i in range(1, n):
        S[i] = S[0]*np.exp(sigma*W[i]+(meu-0.5*(sigma**2))*i/240)
    actual_price = np.array(answer['Close'])
    predicted_price = actual_price[48:]
    Y2 = predicted_price
    return[S,Y2]




for comp_name in stock:
    s = "              DAILY, MONTHLY AND WEEKLY STOCK PRICES for " + comp_name
    fig, axis = plt.subplots(nrows=1, ncols=3,figsize = (15,5),squeeze=False)
    axis[0,0].plot(dailyPrice(comp_name)[0],dailyPrice(comp_name)[1])
    
   
    axis[0,1].plot(weeklyPrice(comp_name)[0],weeklyPrice(comp_name)[1])
    
    axis[0,2].plot(monthlyPrice(comp_name)[0],monthlyPrice(comp_name)[1])
    
    plt.suptitle(s)
    
    plt.show()

for comp_name in stock:
    s = "              DAILY, MONTHLY AND WEEKLY RETURNS for " + comp_name
    fig, axis = plt.subplots(nrows=1, ncols=3,figsize = (15,5),squeeze=False)
    axis[0,0].hist(dailyReturns(comp_name)[0],density=True)
    axis[0,0].plot(dailyReturns(comp_name)[1],dailyReturns(comp_name)[2])
    axis[0,1].hist(weeklyReturns(comp_name)[0],density=True)
    axis[0,1].plot(weeklyReturns(comp_name)[1],weeklyReturns(comp_name)[2])
    axis[0,2].hist(monthlyReturns(comp_name)[0],density=True)
    axis[0,2].plot(monthlyReturns(comp_name)[1],monthlyReturns(comp_name)[2]) 
    plt.suptitle(s)
    plt.show()


for comp_name in stock:
   s = "              DAILY, MONTHLY AND WEEKLY LOG RETURNS for " + comp_name
   fig, axis = plt.subplots(nrows=1, ncols=3,figsize = (15,5),squeeze=False)
   axis[0,0].hist(dailyLogReturns(comp_name)[0],density=True)
   axis[0,0].plot(dailyLogReturns(comp_name)[1],dailyLogReturns(comp_name)[2])
   axis[0,1].hist(weeklyLogReturns(comp_name)[0],density=True)
   axis[0,1].plot(weeklyLogReturns(comp_name)[1],weeklyLogReturns(comp_name)[2])
   axis[0,2].hist(monthlyLogReturns(comp_name)[0],density=True)
   axis[0,2].plot(monthlyLogReturns(comp_name)[1],monthlyLogReturns(comp_name)[2]) 
   plt.suptitle(s)
   plt.show()

print("ORANGE IS ACTUAL STOCK PRICE AND BLUE IS PREDICTED STOCK PRICE")
for comp_name in stock:
    s = "       DAILY AND MONTHLY PREDICTED PRICES AND ACTUAL PRICES for " + comp_name
    fig, axis = plt.subplots(nrows=1, ncols=2,figsize = (15,5),squeeze=False)
    axis[0,0].plot(np.array(range(1,len(predPriceDaily(comp_name)[0])+1)),predPriceDaily(comp_name)[0])
    axis[0,0].plot(np.array(range(1,len(predPriceDaily(comp_name)[1])+1)),predPriceDaily(comp_name)[1])
    axis[0,1].plot(np.array(range(1,len(predPriceMonthly(comp_name)[0])+1)),predPriceMonthly(comp_name)[0])
    axis[0,1].plot(np.array(range(1,len(predPriceMonthly(comp_name)[1])+1)),predPriceMonthly(comp_name)[1])

    plt.suptitle(s)
    plt.show()