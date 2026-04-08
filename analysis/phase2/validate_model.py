import os
import re
import numpy as np

def baremetal_expf(x_arr):
    x = np.clip(x_arr, -15.0, 15.0)
    d = 1.0 + (x / 1024.0)
    d = d ** 1024
    d = np.where(x_arr <= -15.0, 0.0, d)
    d = np.where(x_arr >= 15.0, 3269017.0, d)
    return d

def fann_sigmoid_symmetric(x):
    return -1.0 + (2.0 / (1.0 + baremetal_expf(-x)))

def fann_sigmoid(x):
    return 1.0 / (1.0 + baremetal_expf(-x))

def parse_weights(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    matches = re.findall(r'([-+]?\d*\.\d+)f', content)
    pesos = [float(x) for x in matches]
    
    EXPECTED_WEIGHTS = 961
    if len(pesos) != EXPECTED_WEIGHTS:
        print(f"[AVISO] Esperado {EXPECTED_WEIGHTS} pesos, encontrados {len(pesos)}.")
        
    p = 0
    W1 = np.zeros((50, 16))
    b1 = np.zeros(16)
    for i in range(16):
        if p >= len(pesos): break
        b1[i] = pesos[p]; p += 1
        for j in range(50):
            if p >= len(pesos): break
            W1[j, i] = pesos[p]; p += 1

    W2 = np.zeros((16, 8))
    b2 = np.zeros(8)
    for i in range(8):
        if p >= len(pesos): break
        b2[i] = pesos[p]; p += 1
        for j in range(16):
            if p >= len(pesos): break
            W2[j, i] = pesos[p]; p += 1

    W3 = np.zeros((8, 1))
    b3 = np.zeros(1)
    for i in range(1):
        if p >= len(pesos): break
        b3[i] = pesos[p]; p += 1
        for j in range(8):
            if p >= len(pesos): break
            W3[j, i] = pesos[p]; p += 1
            
    return W1, b1, W2, b2, W3, b3

def predict(X, weights):
    W1, b1, W2, b2, W3, b3 = weights
    H1 = fann_sigmoid_symmetric(X @ W1 + b1)
    H2 = fann_sigmoid_symmetric(H1 @ W2 + b2)
    Out = fann_sigmoid(H2 @ W3 + b3)
    return Out.flatten()
