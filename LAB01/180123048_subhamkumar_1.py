#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math

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

 

M=[1,5,10,20,50,100,200,400]
call_mat=[]
put_mat=[]

def main(m):
  delta=t/m
  delta_1=math.sqrt(delta)
  a=math.exp(sig*delta_1+(r-(sig*sig)/2)*delta)
  b=math.exp(-sig*delta_1+(r-(sig*sig)/2)*delta)
  p=((math.exp(r*delta))-b)/(a-b)
  q=1-p
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
  call_mat.append(sum1)
  put_mat.append(sum2)

    
for k in M:
  main(k)



for i in range(len(M)):
  print("m =",M[i])
  delta_8=t/M[i]
  delta_11=math.sqrt(delta_8)
  aa=math.exp(sig*delta_11+(r-(sig*sig)/2)*delta_8)
  bb=math.exp(-sig*delta_11+(r-(sig*sig)/2)*delta_8)
  if (aa>math.exp(r*delta_8)) and (math.exp(r*delta_8)>bb):
     print(' Call price  =',call_mat[i],',  Put price  =',put_mat[i])
     print(' ')
     print(' ')
  else :
    print('No arbitrage condition violated for m =',M[i])




# In[ ]:




