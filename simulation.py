import numpy as np
variance = 2
# T = 100e-3  # Time span for one slot 100ms
# mu = 0.02   # Sensing duration ratio
# t = mu*T     # Sensing time
Pr = 0.5    # Probability of spectrum occupancy
Pw = -60    # Primary signal power in dBm 
PowerTx = 10**(Pw/10)#*(1e-3)  # Transmitted power 0.1
Nw = -70   # Noise power in dBm -153 or -70
PowerNo = 10**(Nw/10)#*(1e-3) 
g = 10**(-5)  # Path loss coefficeint 10^(-5)
# a = 4 #path loss factor
d = 500#np.array((500, 750, 1000))  # PU-SU distance in meters


def MCS(realize, samples, SU):
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
    return Y, S, meanSNR


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
    H = np.zeros(N)
    H = np.sqrt(-2 * variance * np.log(np.random.rand(N)))/np.sqrt(2)
    H = np.array(H*np.sqrt(g*(d)))  # Fading + path-loss (amplitude loss)
    # f = 500**(-a)
    # H = np.array(H*np.sqrt(f))
    H = np.vstack([H]*samples)
    return H
