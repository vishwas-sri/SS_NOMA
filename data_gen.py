# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:49:05 2023

@author: vishwas
"""
import copy
import numpy as np
import simulation as sm

def main(realize,realize_test, samples, SU):
    realize = realize
    realize_test = realize_test
    samples = samples
    SU = SU
    
# MCS(realize,samples,SU)
    X_train, y_train, _ = sm.MCS(realize, samples, SU)
    X_test, y_test, SNR = sm.MCS(realize_test, samples, SU)
    
    SNR2 = []
    X_test_2 = copy.deepcopy(X_test)
    # for i in range(SU):
    #     # print(SNR[i])
    #     SNR2.append(SNR[i])
    # # SNR = SNR2
    NormSNR = [x/np.sum(SNR) for x in SNR]
    for i in range(SU):
        X_test_2[:,i] = X_test_2[:,i]*NormSNR[i]
        
    np.save('Data\X_train', X_train)
    np.save('Data\y_train', y_train)
    np.save('Data\X_test', X_test)
    np.save('Data\y_test', y_test)
    np.save('Data\X_test_2', X_test_2)
    np.save('Data\SNR', SNR)
    