# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:20:45 2023

@author: vishwas
SS_NOMA
"""
import numpy as np
import MCS
# import matplotlib.pyplot as plt

N = 10000#int(1e5)                    # number of realization
PU = 1                          # Primary user
SU = 2                          # 2 Seconadary user
Pr = 0.5                       # PU active probability
d = 500                         # PU-SU distance
a1 = 0.75                       # power allocation factor
a2 = 0.25                       # ------
eta = 4                         # path loss exponent
samples = 400
realize = 10000

# Generate rayleigh fading coefficient for both users
h1 = np.sqrt(d**(-eta)) * (np.random.randn(1, N) + 1j * np.random.randn(1, N)) / np.sqrt(2)
h2 = np.sqrt(d**(-eta)) * (np.random.randn(1, N) + 1j * np.random.randn(1, N)) / np.sqrt(2)
# h1 = np.vstack([h2])

g1 = np.abs(h1)**2              # gain
g2 = np.abs(h2)**2              # ----

# Transmit power in dBm
Pt = -60

# Transmit power in linear scale
pt = (1e-3) * 10**(Pt/10)

# System bandwidth
BW = int(1e6)

# Noise power (dBm)
No = -174 + 10 * np.log10(BW)

# Noise power (linear scale)
no = (1e-3) * 10**(No/10)

w1 = np.sqrt(no) * (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
w2 = np.sqrt(no) * (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

S = np.random.randint(0, 2, size=(N))
X1 = np.zeros(N)
X2 = np.zeros(N)
for i in range(len(S)):
    if S[i]==1:
        X1[i] = np.random.randn(1) * np.sqrt(pt)
        X2[i] = np.random.randn(1) * np.sqrt(pt)

Y1 = (h1*X1+w1).T
Y2 = (h1*X2+w2).T

y1 = np.abs(Y1)**2/no
y2 = np.abs(Y2)**2/no



# Y_rec, PU_stat, SNR, X_trans = MCS.MonteCarlo(realize, samples, SU, PowerTx=pt, PowerNo=no) 