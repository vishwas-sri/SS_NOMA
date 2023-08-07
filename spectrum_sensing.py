#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:41:48 2023

@author: vishwas

this program determines the Pd and Pf
for different sensing time for all model
"""
import numpy as np
import data_gen as dg
import SS


SU = 3       # No. of SU

realize = 500
realize_test = 50000

Pd = []
Pf = []
samples = 500

# dg.main(realize, realize_test, sample, SU)
result_d, result_f = SS.s_sensing(samples, SU)
    
# Pd = np.array(Pd)
# Pf = np.array(Pf)


 