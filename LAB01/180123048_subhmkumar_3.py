#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math
import matplotlib.pyplot as plt


t=5
sig=0.3
r=0.05
K=105
s0=100
delta=t/20
delta_1=math.sqrt(delta)

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



def pric(i,a,b):
  price_1=s0
  price_2=pow(a,i)*pow(b,20-i)
  price=price_1*price_2
  return price


def combination(n,r):
  c_1=math.factorial(n)
  c_2=math.factorial(n-r)*math.factorial(r)
  c_1=c_1/c_2
  return c_1


aa=math.exp(sig*delta_1+(r-(sig*sig)/2)*delta)
bb=math.exp(-sig*delta_1+(r-(sig*sig)/2)*delta)
if (bb<math.exp(r*t/20)) and (aa>math.exp(r*t/20)):
  print(" There is no arbitrage possible. Proceeding to calculate option prices\n\n\n")
  
else :
  print("There is an arbitrage opportunity possible. The program will terminate\n\n")

M=[0, 2, 4, 6, 12, 18]


def main(m):
  a=math.exp(sig*delta_1+(r-(sig*sig)/2)*delta)
  b=math.exp(-sig*delta_1+(r-(sig*sig)/2)*delta)
  p=((math.exp(r*delta))-b)/(a-b)
  q=1-p
  rem=((20-m)*t)/20
  base=1/math.exp(r*rem)
  print("Present time ", t-rem)
  print("Time remaining ", rem)
  print(" ")
  if m>=0:
    call=[]
    put=[]
    for ben in range(m+1):
      up=ben
      down=m-ben
      s1=0
      s2=0
      for j in range(21-m):
        price_ok=pric(j+ben,a,b)
        price1=aux_call(price_ok)
        price1=price1*combination(20-m,j)
        price1=price1*pow(p,j)*pow(q,20-m-j)
        s1=s1+price1
        price2=aux_put(price_ok)
        price2=price2*combination(20-m,j)
        price2=price2*pow(p,j)*pow(q,20-m-j)
        s2=s2+price2
      s1=s1*base
      s2=s2*base
      call.append(s1)
      put.append(s2)
    
    print("Call option prices\n")
    print(call)
    print("\nPut option prices\n")
    print(put)
    print('\n\n******\n\n')

   
for k in M:
  main(k)


# In[ ]:





# In[ ]:




