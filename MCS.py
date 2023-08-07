# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 11:46:20 2023

@author: vishwas
"""

import numpy as np
from scipy import special

Pr = 0.5    # Probability of spectrum occupancy


eta = 4 #path loss factor
d = 500  # PU-SU distance in meters

        
def MonteCarlo(realize, samples, SU, PowerTx, PowerNo):
    Y = np.zeros((realize, SU))
    S = np.zeros((realize))
    SNR = np.zeros((SU,realize))
 
    noisePower = PowerNo*np.ones(SU)

    for k in range(realize):
        n = gaussianNoise(noisePower, samples)
        H = channel(SU,samples)
        X, S[k] = PUtx(samples, PowerTx, SU)
        PU = np.multiply(H.T, X)
        Z = PU + n
        SNR[:,k] = np.mean(np.abs(PU)**2,axis=1)/noisePower[0]
        Y[k,:] = np.sum(np.abs(Z)**2,axis=1)/(noisePower[0]*samples)

    meanSNR = np.mean(SNR[:,S==1],1)
    meanSNRdB = 10*np.log10(meanSNR)
    return Y, S, meanSNR, X
   
def PUtx(samples, TXPower, N):
    S = 0
    X = np.zeros(samples)
    if (np.random.rand(1) <= Pr):
        S = 1
        X = np.random.randn(samples) * np.sqrt(TXPower)
    X = np.vstack([X]*N)
    return [X, S]


def gaussianNoise(noisePower, samples):
    N = len(noisePower)
    n = np.random.randn(N, samples) * np.sqrt(noisePower[0])
    return n


def channel(N, samples):
    eta = 4
    H = np.zeros(N)
    H = np.sqrt(d**(-eta)) * (np.random.randn(1, N) + 1j * np.random.randn(1, N)) / np.sqrt(2)
    # H = np.sqrt(-2 * variance * np.log(np.random.rand(N)))/np.sqrt(2)
    # H = np.array(H*np.sqrt(g*(d)))  # Fading + path-loss (amplitude loss)
    # f = 500**(-a)
    # H = np.array(H*np.sqrt(f))
    H = np.vstack([H]*samples)
    return H