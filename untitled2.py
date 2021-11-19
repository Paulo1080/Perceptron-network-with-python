# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:41:37 2021

@author: Paulo_Vi
"""

import pandas

def propagacao (W, T, x):
    u = 0
    for i in range(len(W)):
        u += W.iloc[i, 0] * x[i]
    u += T
    return u

def ativacao (u):
    return 1 if (u >0) else 0

def atualizaPesos(W, T, n, e, x):
    for i in range(len(W)):
        W.iloc[i, 0] += n * e * x[i]
        
    T += n * e
    return (W, T)
        
#definir os parametros de entrada
n = 0.5
W_INI = 0.1
NUM_EPOCAS = 100

#ler a base de dados
db = pandas.read_csv('base.txt', header=None)
data = db.iloc[: , :len(db.iloc[0, :])-1]
classes = db.iloc[: , len(db.iloc[0, :])-1]

T = W_INI
W = pandas.DataFrame([W_INI for a in range(len(data.iloc[0, :]))])

####################TREINAMENTO#############################

print("="*50)
print("TREINAMENTO")
print("="*50)

for epoca in range(1,(NUM_EPOCAS+1)):
    
    sse = 0.0
    for i in range(len(data)):
        x = data.iloc[i, :]
        d = classes[i]
        u = propagacao(W,T,x)
        y = ativacao(u)
        e = d - y
            
        if e != 0:
            W, T = atualizaPesos(W, T, n, e, x)
            sse += e**2
                
    sse /= len(data)
    
    print('epoca:', epoca, '\tsse:', sse)
    
    if(sse == 0):
        break
    
print()
print(W)                       
print(T)            
            
         ####################TESTE#############################

print("="*50)
print("TESTE")
print("="*50)

sse = 0.0
for i in range(len(data)):
    x = data.iloc[i, :]
    d = classes[i]
    u = propagacao(W,T,x)
    y = ativacao(u)
    e = d - y
            
    if e != 0:
        sse += e**2
                
sse /= len(data)
    
print('epoca:', epoca, '\tsse:', sse)
    

            

