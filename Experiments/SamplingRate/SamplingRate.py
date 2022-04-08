# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:10:53 2022

@author: alexa
"""

import sys
sys.path.insert(0, "E:\GitHub\DigitalTwinInjectionMolding")
# sys.path.insert(0, 'C:/Users/rehmer/Documents/GitHub/DigitalTwinInjectionMolding/')

import pickle as pkl  
import matplotlib.pyplot as plt   
import numpy as np

from scipy.fft import fft, fftfreq, ifft


from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y




c1 = pkl.load(open('E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/cycle1.pkl','rb'))


# path = '/home/alexander/Downloads/Versuchsplan/' # @work
path = 'E:/GitHub/DigitalTwinInjectionMolding/data/HighFrequencyMeasurements/'

file = 'cycle821.pkl'
file = 'cycle828.pkl'


data = pkl.load(open(path+file,'rb'))

x = data['meas'].values

#  Remove the mean to get rid of the DC component
# x = x-np.mean(x)

# No windowing needed, because a transient signal that fits entirely into the
# time record is measured

N = len(x)
dt = data.index[1]-data.index[0]
fs = 1/dt

cutoff = fs/2

x_filt = butter_lowpass_filter(x,25,fs)
data['meas_filt'] = x_filt

# x=x_filt

X = fft(x)/N                                                                    # Spectrum
df = fftfreq(N,dt)                                                              # Frequency resolution

X_A = abs(X)                                                                    # Amplitude Spectrum
X_dB = 20*np.log10(X_A)                                                         # Amplitude Spectrum in dB
P = X*np.conjugate(X)/N

X_red = X*N

bin_cutoff = 410

X_red[bin_cutoff:N//2]=0
X_red[N//2:N-bin_cutoff+1]=0

df[bin_cutoff]
df[N-bin_cutoff]

x_reconstruct = ifft(X_red).real

plt.close('all')

plt.figure()
plt.plot(data['meas']) 
plt.plot(data['meas_filt'])
plt.plot(c1['p_inj_ist'])

plt.figure()
plt.plot(df[1:N//2],X_dB[1:N//2],'x')

plt.figure()
plt.plot(x)
plt.plot(x_reconstruct)



