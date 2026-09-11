import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft2, fftshift
from scipy.interpolate import RegularGridInterpolator

# 2. Trace Normalization (AGC-like)
def normalize_traces(data):
    # Normalize each trace by its max to see distant arrivals
    norm = np.max(np.abs(data), axis=1, keepdims=True)
    return data / (norm + 1e-10)

def plot_wiggle(xt, dt, skip=3, scale=1.5, normalized=False, fill=True, landscape='horizontal', fig=None, ax=None, color='black'):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    t = np.arange(xt.shape[1]) * dt

    if normalized:
        xt = normalize_traces(xt)

    if landscape == 'horizontal':
        for i in range(0, xt.shape[0], skip):
            trace = xt[i, :] * scale + i
            ax.plot(t, trace, color=color, lw=0.5)
            if fill:
                ax.fill_between(t, i, trace, where=(trace > i), color=color, alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel Index')
        ax.set_title('Seismic Record Section (Wiggle Plot)')
        ax.invert_yaxis()
    elif landscape == 'vertical':
        for i in range(0, xt.shape[0], skip):
            trace = xt[i, :] * scale + i
            ax.plot(trace, t, color=color, lw=0.5)
            if fill:
                ax.fill_betweenx(t, i, trace, where=(trace > i), color=color, alpha=0.5)
        ax.set_ylabel('Time (s)')
        ax.set_xlabel('Channel Index')
        ax.set_title('Seismic Record Section (Wiggle Plot)')
        ax.invert_yaxis()
    return fig, ax


# 4. f-k (Frequency-Wavenumber) Analysis
# from labseis.visual import plot_fk
def plot_fk(data, fs, dx, pad_spatial=256):
# 1. Compute f-k spectrum with spatial padding
    n_spatial = pad_spatial if pad_spatial > data.shape[0] else data.shape[0]
    n_temporal = data.shape[1]
    
    # Apply 2D FFT with padding in the first (spatial) dimension
    fk = fftshift(fft2(data, s=(n_spatial, n_temporal)))
    fk_abs = 10 * np.log10(np.abs(fk)**2)
    
    freqs = fftshift(np.fft.fftfreq(n_temporal, 1/fs))
    k_vals = fftshift(np.fft.fftfreq(n_spatial, dx))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(k_vals, freqs, fk_abs.T, cmap='magma', shading='auto')
    ax.set_ylim(0, 15000) # Focus on signal band
    ax.set_xlabel('Wavenumber (1/m)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('f-k Spectrum (Apparent Velocity Analysis)')
    plt.colorbar(im, label='Power (dB)')
    return fig, ax

def plot_fv(data, fs, dx, v_range=(10, 2000), num_v=300, pad_spatial=256):
    """
    Plots Frequency vs Apparent Velocity by remapping the f-k spectrum.
    v = f / k. Includes spatial padding to improve wavenumber resolution.
    """
    # 1. Compute f-k spectrum with spatial padding
    n_spatial = pad_spatial if pad_spatial > data.shape[0] else data.shape[0]
    n_temporal = data.shape[1]
    
    # Apply 2D FFT with padding in the first (spatial) dimension
    fk = fftshift(fft2(data, s=(n_spatial, n_temporal)))
    fk_abs = np.abs(fk)**2
    
    freqs = fftshift(np.fft.fftfreq(n_temporal, 1/fs))
    k_vals = fftshift(np.fft.fftfreq(n_spatial, dx))
    
    # 2. Focus on positive frequencies
    f_mask = freqs >= 0
    f_pos = freqs[f_mask]
    fk_pos = fk_abs[:, f_mask]
    
    # 3. Create interpolator for (k, f) -> power
    interp = RegularGridInterpolator((k_vals, f_pos), fk_pos, bounds_error=False, fill_value=0)
    
    # 4. Define target Velocity-Frequency grid
    v_axis = np.linspace(v_range[0], v_range[1], num_v)
    V, F = np.meshgrid(v_axis, f_pos)
    
    # 5. Map (v, f) to k: k = f/v
    K_target = F / V
    
    # 6. Interpolate power onto v-f grid
    fv_abs = interp((K_target, F))
    fv_db = 10 * np.log10(fv_abs + 1e-15)
    
    # 7. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(v_axis, f_pos, fv_db, cmap='magma', shading='auto')
    ax.set_xlabel('Apparent Velocity (m/s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'f-v Spectrum (Spatial Pad: {n_spatial})')
    ax.set_ylim(0, 15000) 
    plt.colorbar(im, label='Power (dB)')
    return fig, ax

# 5. Improved Spectrogram (dB scale)
def plot_spectrogram_db(trace, fs, ch=0, fig=None):
    f, t, sxx = sig.spectrogram(trace, fs, nperseg=512, noverlap=256)
    if fig is None:
        fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])       # top row spans both columns (larger)
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)       # bottom-left small
    T = np.arange(len(trace)) / fs
    ax1.plot(T, trace, label=f'ch{ch}')
    ax1.set_title(f'Time Series (Channel {ch})')
    ax1.set_ylabel('Velocity (mm/s)')

    im = ax2.pcolormesh(t, f, 10 * np.log10(sxx), shading='gouraud', cmap='magma')
    ax2.set_yscale('log')
    ax2.set_ylim(100, fs/2)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    plt.colorbar(im, label='dB')
    return fig


# 6. Virtual Source Gather (Cross-Correlation)
from labseis import signal

def plot_virtual_source(data, ref_idx, dt):
    corr, lags = signal.sweep_crosscorrelation_core(data, data[ref_idx], dt)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, aspect='auto', extent=[lags[0], lags[-1], data.shape[0], 0], 
                   cmap='seismic', vmin=-0.1, vmax=0.1)
    ax.set_xlim(-0.005, 0.005)
    ax.set_xlabel('Lag Time (s)')
    ax.set_ylabel('Channel Index')
    ax.set_title(f'Virtual Source Gather (Ref Channel: {ref_idx})')
    plt.colorbar(im)
    return fig, ax


import matplotlib.gridspec as gridspec

def plot_wavefield_image_with_traces(data, time, channels_to_plot=[15, 22], time_range=None, vmin=-0.01, vmax=0.01):
    """
    Plots a wavefield heatmap with selected channel traces on top.
    """
    if time_range is not None:
        idx = np.where((time >= time_range[0]) & (time <= time_range[1]))[0]
    else:
        idx = np.arange(len(time))
        
    ch_count = data.shape[0]
    ch_axis = np.arange(ch_count)
    
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
    
    # Top panel: Selected traces
    ax1 = fig.add_subplot(gs[0])
    for ch in channels_to_plot:
        ax1.plot(time[idx], data[ch, idx], label=f'ch{ch}')
        ax1.set_title('Selected Traces')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Amplitude')
    
    # Bottom panel: Heatmap
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    im = ax2.imshow(data[:, idx], aspect='auto', 
                    extent=[time[idx][0], time[idx][-1], ch_axis[-1], ch_axis[0]], 
                    cmap='seismic', vmin=vmin, vmax=vmax)
    # im = ax2.contourf(time[idx], ch_axis, data[:, idx], cmap='seismic', vmin=vmin, vmax=vmax)
    ax2.set_ylabel('Channel Index')
    ax2.set_xlabel('Time (s)')
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def wiggles(xt, t_axis, offset, fig=None, ax=None, scalebar=True, scale=0.5, scale_range='trace', legends=None):
    if ax==None:
        fig, ax = plt.subplots(figsize=(10, 5))

    if scale_range=='trace':
        normalized = np.max(xt, axis=-1)

    elif scale_range == 'all':
        normalized = np.max(xt, axis=-1)
        normalized[normalized<np.max(normalized)] = np.max(normalized)

    for i in range(xt.shape[0]):
        if legends is not None:
            ax.plot(t_axis, scale*xt[i]/normalized[i] + offset[i],label=legends[i])
        else:
            ax.plot(t_axis, scale*xt[i]/normalized[i] + offset[i])


    if scalebar:
        for i in range(xt.shape[0]):
            ax.plot([t_axis[0],t_axis[0]], [offset[i],offset[i]+scale], 'k')
            ax.annotate('{}'.format(normalized[i]), xy=[0,offset[i]+0.1])

    plt.legend(loc = 'upper right')
    # if legends is not None:
    #     for i in range(xt.shape[0]):
    #         ax.annotate('{}'.format(legends[i]), xy=[t_axis[-1],offset[i]+0.1])
    

    return fig, ax


def plot_waveform(time_axis, data, title, fig=None, ax=None, label=None, xlabel="Time (seconds)", ylabel="Velocity (mm/s)"):
    # Plot the 1C waveform data
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_axis, data, label=label if label else f'{title}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid()
    return ax

def plot_psd(data, sample_frequency, title, fig=None, ax=None, label=None):
    # Plot the PSD
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    f, Pxx = sig.welch(data, fs=sample_frequency, nperseg=10000)
    ax.plot(f.T/1000, 10 * np.log10(Pxx), label=label if label else f'{title}')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_title(f'PSD: {title}')
    ax.grid()
    return fig, ax