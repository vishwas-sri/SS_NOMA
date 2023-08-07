"""
This is the main file that perform
spectrum sensing
"""

import numpy as np

from model import Classification
import matplotlib.pyplot as plt
import plot


# Simulation Parameters
# T = 100e-3  # Time span for one slot 100ms
# mu = 0.02   # Sensing duration ratio
# t = mu*T    # Sensing time
# fs = 50e3
# PU = 1       # No. of PU
# SU = 3       # No. of SU
# Pr = 0.5    # Probability of spectrum occupancy
# # Pd = 0.9    # Probability of detection
# # Pf = 0.1    # Probability of false alarm
# m = np.full(SU, 20)      # Battery capacity
# Eh = 0.1    # Harvested energy during one slot
# # Pw = -60    # Primary signal power in dBm
# # PowerTx = 10**(Pw/10)  # Transmitted power
# # Nw = -70    # Noise power in dBm
# # PowerNo = 10**(Nw/10)
# # g = 10**-5  # Path loss coefficeint 10^(-5)
# # d = 500 # PU-SU distance in meters
# samples = 350 #int(2*t*fs)
# # samples = sample.astype(int) # No. of sample per sensing time
# # w = 5e6     # Bandwidth
# # samples = 50  # No. of sample
# # N = SU
# realize = 500
# realize_test = 50000

# th=[]

def s_sensing(samples, SU):
    X_train = np.load('Data\X_train.npy')
    y_train = np.load('Data\y_train.npy')
    X_test = np.load('Data\X_test.npy')
    y_test = np.load('Data\y_test.npy')
    X_test_2 = np.load('Data\X_test_2.npy')
    SNR = np.load('Data\SNR.npy')
    
    file = []
    
    
    demo =Classification(X_train=X_train,y_train=y_train,X_test=X_test,
                        y_test=y_test, samples=samples,SU=SU, X_test_2=X_test_2)
    
    
    file.append(demo.Linear_SVM())
    file.append(demo.Gaussian_SVM())
    file.append(demo.Logistic())
    
    file.append(demo.S1())
    file.append(demo.S2())
    file.append(demo.S3())
    file.append(demo.OR())
    file.append(demo.AND())
    file.append(demo.MRC())
    
    desired_pd = 0.9
    desired_pf = 0.1
    fpr = []
    tpr = []
    pd = []
    pf = []
    for i in range(len(file)):
        fpr,tpr,_,_ = file[i]
        idx = np.abs(tpr - desired_pd).argmin()
        # Calculate PD and PF using the threshold
        pd.append(tpr[idx])
        pf.append(fpr[idx])
    
    # Print the results
    # print('PD:', pd)
    # print('PF:', pf)
    
    # total = len(y_test)
    
    # PH1 = (np.sum(y_test)/total)
    # PH0 = 1-PH1
    
    # for y_a, y_p in zip(y_test, y_p):
    #     if y_a == 1 and y_p == 1:
    #         Pd += 1
    #     if y_a == 0 and y_p == 1:
    #         Pf += 1
    
    # Pd = Pd/total
    # Pf = Pf/total
    # one_pd = W/total
    
    # th1 = (T-t)/T
    # th=th1*(PH1*one_pd+PH0*one_pf)
    # th.append(th1*(PH0*one_pf))
    # print(th)
    
    file.sort(key=lambda x:x[2],reverse=True)
    if file:
        plot.show_plot(file)  # , mark
    plt.show()
    
    
    # print('Sample ',j,'completed')
        
    # plt.plot(th)
    # plt.grid(True)
    # plt.show()
    return pd,pf

