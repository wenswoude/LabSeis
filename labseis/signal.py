
import scipy as sp
import scipy.signal as sig
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d

import numpy as np
import matplotlib.pyplot as plt


def bandpass(d_xt, dt, fmin, fmax, order=2):
    '''
    Apply a zero-phase bandpass filter to an array on time domain.
    '''
    fnyq = 0.5/dt
    f0 = fmin/fnyq
    f1 = fmax/fnyq
    wn = [f0, f1]
    b, a = sp.signal.butter(order, wn, 'bandpass')
    d_xt = sp.signal.filtfilt(b, a, d_xt,  axis=- 1)
    return d_xt


def spectrogram(x, fs, tstart=0, plot=False, axes=None, **kwargs):
    f, t, Sxx = sp.signal.spectrogram(x, fs, return_onesided=True, **kwargs)
    # f = sp.fft.fftshift(f)
    # Sxx =  sp.fft.fftshift(Sxx, axes=0)

    if plot:
        if axes==None:
            fig, axes = plt.subplots()

        axes.pcolormesh(t+tstart, f, 10*np.log10(Sxx), shading='gouraud')
        axes.set_ylabel('Frequency [Hz]')
        axes.set_xlabel('Time [sec]')
        plt.show()
    return f, t, Sxx

def sweep_crosscorrelation_core(x_xt, y, dt, alpha=0):
    '''
    Cross correlation of x and y

                    x(w) * conj(y(w)) * G(w)
    ---------------------------------------------------------
    sqrt{x_smoothed(w)*conj(x_smoothed(w))}* sqrt{y_smoothed(w)*conj(y_smoothed(w)]}

    0, preprocessing: demean, detrend, tapering
    1, Fourier transform
    2, cross correlation (normalized) = cross coherence

    x <numpy.ndarray>: space-time domian seismic data, 2D matrix 
    y <numpy.ndarray>: time series seismic data from certain channel, 1D
    dt <fload>: time domain sampling step = 1/sampling rate
    alpha <float>: gaussian low pass filter based on G = exp(-f^2/alpha^2), f (unit: Hz)
    t_nor <str> : time domain normalization: '1-bit' or None
    
    This function is based on IRIS summer courses ROSES 2020.
    '''
    ns_t = x_xt.shape[1]    
    ns_x = x_xt.shape[0]
    #1. Demean
    x_xt = sp.signal.detrend(x_xt, type='constant')
    y = sp.signal.detrend(y, type='constant')
    #2. Detrend
    x_xt = sp.signal.detrend(x_xt, type='linear')
    y = sp.signal.detrend(y, type='linear')

    # running-means normalization
    # eqfmin=0.5 
    # eqfmax = 30
    # eqtempx=obspy.signal.filter.bandpass(np.array(x),freqmin=eqfmin,freqmax=eqfmax,
    #                                      df=1/dt,corners=4,zerophase=True)
    # eqtempy=obspy.signal.filter.bandpass(np.array(y),freqmin=eqfmin,freqmax=eqfmax,
    #                                      df=1/dt,corners=4,zerophase=True)
    # eqsmoothNUM=int(2/dt) #get samples to smooth over..
    # # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    # eq_smoothx = np.ones(x.shape)
    # for i in range(ns_x):
    #     eq_smoothx[i,:] = np.convolve(np.abs(eqtempx[i,:]), np.ones(eqsmoothNUM)/eqsmoothNUM, mode='same')
    # eq_smoothy = np.convolve(np.abs(eqtempy), np.ones(eqsmoothNUM)/eqsmoothNUM,mode='same')
    # eq_smooth=np.sqrt(np.mean(eq_smoothx*eq_smoothy))
    # x= x/(eq_smooth)
    # y= y/(eq_smooth)
    
    # time domain normalization by deviding envolope
    x_env = np.abs(sp.signal.hilbert(x_xt))
    y_env = np.abs(sp.signal.hilbert(y))
    # tapering
    tukeytaper = sp.signal.windows.tukey(ns_t, alpha=alpha, sym=True)
    x_xt = x_xt/x_env*tukeytaper
    y = y/y_env*tukeytaper
          
    ########################## FFT of data
    npts = ns_t
    fx   = np.fft.fft(x_xt, npts, axis=-1)
    fy   = np.fft.fft(y, npts, axis=-1)
    freq = np.fft.fftfreq(npts, dt)
    
    Px   = abs(fx)**2
    Py   = abs(fy)**2

    # Apply a smoothing operator, improve variance. 
    Px_smooth = np.ones(Px.shape)
    nsmooth = 21
    for i in range(ns_x):
        Px_smooth[i,:] = np.convolve(Px[i,:], np.ones(nsmooth)/nsmooth)[0:npts]

    Py_smooth = np.convolve(Py, np.ones(nsmooth)/nsmooth)[0:npts]

    ########################## The cross spectrum
    Sxy  = np.conj(fy)*fx
    
    # G    = np.exp(-freq**2/alpha**2)
    ########################## Normalized Cross correlation! 
    cohe = Sxy/(np.sqrt(Px_smooth)*np.sqrt(Py_smooth)+ 0.0001*np.max(np.sqrt(Px)*np.sqrt(Py)))
    # cohe = Sxy/(np.sqrt(Px)*np.sqrt(Py) + 0.0001*np.max(np.sqrt(Px)*np.sqrt(Py)))
    xycorr = np.real(sp.fft.ifft(cohe, axis=-1))
    xycorr = np.fft.ifftshift(xycorr, axes=-1)
    
    # Delay time vector, same as above.. only run on the first round
    nf   = int((npts+1)/2)  # window length 
    T    = np.linspace(-nf*dt, nf*dt, npts)

    return xycorr, T

def sweep_crosscorrelation_timedomain(x_xt, y, dt):
    '''
    
    '''
    ns_t = x_xt.shape[1]    
    ns_x = x_xt.shape[0]
    #1. Demean
    x_xt = sp.signal.detrend(x_xt, type='constant')
    y = sp.signal.detrend(y, type='constant')
    #2. Detrend
    x_xt = sp.signal.detrend(x_xt, type='linear')
    y = sp.signal.detrend(y, type='linear')
          
    # Time domain cross correlation:
    
    npts_y = len(y)
    xycorr = np.empty(x_xt.shape)
    
    for i in range(ns_x):
        xycorr[i,:] = np.convolve(x_xt[i,:], np.flip(y))[npts_y-1:]
    
    # Delay time vector, same as above.. only run on the first round
    T    = np.linspace(0, ns_t*dt, ns_t)

    return xycorr, T

def crosscorrelation_for_timeshift(x_xt, y, dt):
    ns_t = x_xt.shape[1]    
    ns_x = x_xt.shape[0]
    #1. Demean
    x_xt = sp.signal.detrend(x_xt, type='constant')
    y = sp.signal.detrend(y, type='constant')
    #2. Detrend
    x_xt = sp.signal.detrend(x_xt, type='linear')
    y = sp.signal.detrend(y, type='linear')
          
    # Time domain cross correlation:
    npts_y = len(y)
    xycorr = np.empty([ns_x, 2*ns_t-1])
    
    for i in range(ns_x):
        xycorr[i,:] = np.convolve(x_xt[i,:], np.flip(y))
    
    # Delay time vector, same as above.. only run on the first round
    T    = np.arange(-ns_t+1, ns_t, 1)*dt

    return xycorr, T

# def get_piezo_to_velocity_transfer_function(velocity_data, piezo_data, fs):
#     # Compute transfer function H(f) from calibration data
#     v = np.asarray(velocity_data).flatten()
#     p = np.asarray(piezo_data).flatten()
#     V = np.fft.rfft(v)
#     P = np.fft.rfft(p)
#     H = V*P / (P*P + 1e-6)
#     freqs = np.fft.rfftfreq(len(v), 1/fs).flatten()
#     return H.squeeze(), freqs.squeeze(), fs

# def get_piezo_to_velocity_transfer_function_cross_spectrum(velocity_data, piezo_data, fs, eps=1e-6):
#     # Compute transfer function H(f) from calibration data using cross-spectrum method
#     v = np.asarray(velocity_data).flatten()
#     p = np.asarray(piezo_data).flatten()
#     V = np.fft.rfft(v)
#     P = np.fft.rfft(p)
#     S_vp = V * np.conj(P)
#     S_pp = P * np.conj(P)
#     H = S_vp / (S_pp + eps)
#     freqs = np.fft.rfftfreq(len(v), 1/fs).flatten()
#     return H.squeeze(), freqs.squeeze(), fs

def get_piezo_to_velocity_transfer_function_cross_spectrum_welch(velocity_data, piezo_data, fs, nperseg=480000, eps=1e-6):
    # Compute transfer function H(f) from calibration data using cross-spectrum method with Welch's method
    v = np.asarray(velocity_data).flatten()
    p = np.asarray(piezo_data).flatten()
    f, S_vp = sig.csd(v, p, fs=fs, nperseg=nperseg)
    f, S_pp = sig.welch(p, fs=fs, nperseg=nperseg)
    H = S_vp / (S_pp + eps)
    return H.squeeze(), f.squeeze(), fs



def apply_transfer_function(new_piezo, H, freqs_H, fs_H, fs_new, filter=True, band=(100, 100000), order=4):
    # Resample new_piezo if needed
    if fs_new != fs_H:
        num_samples = int(len(new_piezo) * fs_H / fs_new)
        new_piezo = resample(new_piezo, num_samples)

    # FFT of new data
    P_new = np.fft.rfft(new_piezo)
    freqs_new = np.fft.rfftfreq(len(new_piezo), 1/fs_H)

    # Interpolate H to new frequency bins
    H_interp = interp1d(freqs_H, H, bounds_error=False, fill_value='extrapolate')
    H_new = H_interp(freqs_new)

    # Apply transfer function
    V_est = P_new * H_new

    # Inverse FFT to get estimated velocity in time domain  
    velocity_estimated = np.fft.irfft(V_est, n=len(new_piezo))

    # Optional bandpass filter
    if filter:
            b, a = butter(order, [band[0]/(fs_new/2), band[1]/(fs_new/2)], btype='band')
            velocity_estimated = filtfilt(b, a, velocity_estimated)

    return velocity_estimated

# def smooth_transfer_function(H, method='gaussian', width=5):
#     """
#     Smooth the transfer function H(f) in the frequency domain.
#     method: 'gaussian' or 'moving_average'
#     width: smoothing window width (in frequency bins)
#     """
#     if method == 'gaussian':
#         H_smooth = gaussian_filter1d(H.real, width) + 1j * gaussian_filter1d(H.imag, width)
#     elif method == 'moving_average':
#         def moving_average(x, w):
#             return np.convolve(x, np.ones(w)/w, mode='same')
#         H_smooth = moving_average(H.real, width) + 1j * moving_average(H.imag, width)
#     else:
#         raise ValueError('Unknown smoothing method')
#     return H_smooth


# def smooth_transfer_function_gaussian(H, sigma=2):
#     """
#     Smooth the transfer function H using a Gaussian filter.
#     sigma: standard deviation for Gaussian kernel (in frequency bins)
#     """
#     H_amp = np.abs(H)
#     H_phase = np.angle(H)
#     H_amp_smooth = gaussian_filter1d(H_amp, sigma)
#     H_phase_smooth = gaussian_filter1d(H_phase, sigma)
#     H_smooth = H_amp_smooth * np.exp(1j * H_phase_smooth)
#     return H_smooth