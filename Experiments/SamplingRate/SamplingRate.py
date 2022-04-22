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
from scipy.signal.windows import flattop, hann, hamming

from scipy.signal import butter, lfilter, freqz, filtfilt
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def zero_phase_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# fir filter  100 koeff
# iir filter
# nullphasenfilter filtfilt


c1 = pkl.load(open('E:/GitHub/DigitalTwinInjectionMolding/data/Versuchsplan/cycle1.pkl','rb'))


# path = '/home/alexander/Downloads/Versuchsplan/' # @work
path = 'E:/GitHub/DigitalTwinInjectionMolding/data/HighFrequencyMeasurements/'

file = 'cycle821.pkl'
file = 'cycle828.pkl'

data = pkl.load(open(path+file,'rb'))

x = data['meas'].values

#  Remove the mean to get rid of the DC component
x = x-np.mean(x)
data['meas_centered'] = x


# Calculate some quantities
N = len(x)
dt = data.index[1]-data.index[0]
fs = 1/dt

# Apply a windows as not the full cycle is measured
window_hann = hann(N,sym=True)
window_hamming = hamming(N,sym=True)
window_flat = flattop(N,sym=True)

x_window = x*window_hamming


X = fft(x_window)/N                                                                    # Spectrum
df = fftfreq(N,dt)                                                              # Frequency resolution

X_A = abs(X)                                                                    # Amplitude Spectrum
X_dB = 20*np.log10(X_A)                                                         # Amplitude Spectrum in dB
P = (X*np.conjugate(X)/N).real

bin_25 = 205






print('Ratio of Amplitudes under 25 Hz: ' + str( sum(X_A[1:bin_25])/sum(X_A[1:N//2])) )
print('Ratio of Power under 25 Hz: ' + str( sum(P[1:bin_25])/sum(P[1:N//2])) )

amplitude_ratio = [sum(X_A[1:bin_i])/sum(X_A[1:N//2]) for bin_i in range(0,N//2)]
power_ratio = [sum(P[1:bin_i])/sum(P[1:N//2]) for bin_i in range(0,N//2)]


# plt.figure()
fig,ax = plt.subplots(2,1)
ax21 = ax[0].twinx()
ax22 = ax[1].twinx()

ax[0].plot(df[1:N//2],X_dB[1:N//2],'x')
ax21.plot(df[1:N//2],amplitude_ratio[1:N//2],'r')

ax[1].plot(df[1:N//2],P[1:N//2],'x')
ax22.plot(df[1:N//2],amplitude_ratio[1:N//2],'r')

ax[0].set_xlim(0,25)
ax[1].set_xlim(0,25)

ax[0].set_title('Amplitude Sepctrum (dB)')
ax[1].set_title('Power Spectrum')

ax[1].set_xlabel('Hz')



# cutoff = fs/2

''' Filter Signal with zero phase filter and cut off frequency 25 Hz'''
x_filt = zero_phase_filter(data=x,cutoff=25,fs=fs,order=16)
data['meas_filt'] = x_filt

# plt.close('all')

plt.figure()
plt.plot(data['meas_centered']) 
plt.plot(data['meas_filt'])

# Reconstruction seems to be reasonable good


''' Delete coefficients associated with fequencies above 25 Hz and apply inverse
fft '''

X_red = X*N

X_red[0]=0
X_red[bin_25:N//2]=0
X_red[N//2:N-bin_25+1]=0

x_reconstruct = ifft(X_red).real

plt.figure()
plt.plot(x)
plt.plot(x_window)
plt.plot(x_reconstruct)