#!/usr/bin/env python
# coding: utf-8

# In[9]:


import math
import matplotlib.pyplot as plt
import numpy as np



t=5
sig=0.3
r=0.05
K=105
s0=100

def aux_put(pr):
  if (pr<=K):
    return (K-pr)
  elif (pr>K):
    return 0


def aux_call(pr):
  if (pr>=K):
    return (-K+pr)
  elif (pr<K):
    return 0



def pric(i,a,b,m):
  price_1=s0
  price_2=pow(a,i)*pow(b,m-i)
  price=price_1*price_2
  return price


def combination(n,r):
  c_1=math.factorial(n)
  c_2=math.factorial(n-r)*math.factorial(r)
  c_1=c_1/c_2
  return c_1


print("We have taken m ranging from 1 to 100 in both the cases\n\n\n")


M=list(range(1, 101))


M_2=np.arange(1, 106, 5).tolist()


call=[]
put=[]
call_1=[]
put_1=[]

def main(m,opt):
  delta=t/m
  delta1=math.sqrt(delta)
  a=math.exp(sig*delta1+(r-(sig*sig)/2)*delta)
  b=math.exp(-sig*delta1+(r-(sig*sig)/2)*delta)
  p=((math.exp(r*delta))-b)/(a-b)
  q=1-p
  if (a>math.exp(r*t/m)) and (b<math.exp(r*t/m)):
    base=1/(math.exp(r*t))
    sum1=0
    sum2=0
    for i in range(m+1):
     price_o=pric(i,a,b,m)
     price=aux_call(price_o)
     price=price*combination(m,i)
     price=price*pow(p,i)*pow(q,m-i)
     sum1=sum1+price
     price_o=pric(i,a,b,m)
     price=aux_put(price_o)
     price=price*combination(m,i)
     price=price*pow(p,i)*pow(q,m-i)
     sum2=sum2+price
    sum1=sum1*base
    sum2=sum2*base
    if opt==1 :
     call.append(sum1)
     put.append(sum2)
    if opt==2 :
     call_1.append(sum1)
     put_1.append(sum2)
  
  else :
    print('No arbitrage condition is violated, calculation terminated for m=',m)
 

    
for k in M:
  main(k,1)

plt.plot(M,call,label='call', color="r")
plt.plot(M,put,label='put',color="g")
plt.xlabel('m, increased by +1 at a time')
plt.ylabel('Price')
plt.legend(loc="upper right")
plt.show()

print('\n\n')
    
for k in M_2:
  main(k,2)

plt.plot(M_2,call_1,label='call', color="r")
plt.plot(M_2,put_1,label='put',color="g")
plt.xlabel('m, increased by +5 at a time')
plt.ylabel('Price')
plt.legend(loc="upper right")
plt.show()


# In[ ]:





# In[ ]:




