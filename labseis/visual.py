import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

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