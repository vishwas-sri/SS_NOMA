# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:45:54 2023

@author: vishwas
"""

import numpy as np
import data_gen as dg
import SS
import simulation as sm
from model import Classification


SU = 3       # No. of SU
Eh = 0.1
k = 2
delta = 3
Es = Eh/k
Eth = delta*k*Es
m = 20

realize = 500
realize_test = 1000

Pd = []
Pf = []
samples = 500

X_train, y_train, _ = sm.MCS(realize, samples, SU)
X_test, y_test, SNR = sm.MCS(realize_test, samples, SU)
# result_d, result_f = SS.s_sensing(samples, SU)
file =[]
for i in range(1):
    if Eh<=Eth:
        demo =Classification(X_train=X_train,y_train=y_train,X_test=X_test,
                            y_test=y_test, samples=samples,SU=SU, X_test_2=None)
        ypred = demo.Linear_SVM()
# ypred = ypred[0]