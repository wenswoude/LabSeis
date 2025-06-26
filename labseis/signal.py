
import scipy as sp
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