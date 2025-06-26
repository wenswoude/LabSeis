import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt


def refine_argmax(corr, lags):
    """
    Refine the maximum of a cross-correlation function using quadratic (parabolic) interpolation.
    
    Parameters:
        corr (np.ndarray): 1D cross-correlation array.
        lags (np.ndarray): 1D array of lag values corresponding to the cross-correlation.
        
    Returns:
        refined_lag (float): The refined (sub-sample) lag at which the cross-correlation peaks.
    """
    max_idx = np.argmax(corr)
    
    # If the maximum is at a boundary, return the discrete lag value.
    if max_idx == 0 or max_idx == len(corr) - 1:
        return lags[max_idx]
    
    # Get the cross-correlation values at the maximum and its immediate neighbors.
    y_m1 = corr[max_idx - 1]
    y_0  = corr[max_idx]
    y_p1 = corr[max_idx + 1]
    
    # Compute the denominator for the quadratic interpolation.
    denominator = (y_m1 - 2 * y_0 + y_p1)
    if denominator == 0:
        delta = 0.0
    else:
        # Compute the sub-sample offset (delta).
        delta = 0.5 * (y_m1 - y_p1) / denominator
    
    # Compute the refined lag (discrete lag + sub-sample correction).
    refined_lag = lags[max_idx] + delta
    return refined_lag

def crosscorr_delay(ref_win, seg_win, fs=1.0):
    """
    Compute the time delay between two windowed segments using cross-correlation.
    
    Parameters:
        ref_win (np.ndarray): Window of the reference signal.
        seg_win (np.ndarray): Window of the segment signal.
        fs (float): Sampling frequency in Hz.
        
    Returns:
        dt (float): Delay (in seconds) computed from the cross-correlation peak.
    """
    # Compute cross-correlation between the two windows.
    corr = correlate(seg_win, ref_win, mode='full')
    lags = np.arange(-len(ref_win) + 1, len(ref_win))
    # Use quadratic interpolation to refine the delay.
    lag_max = refine_argmax(corr, lags)
    dt = lag_max / fs  # convert lag (in samples) to time (seconds)
    return corr, dt

def window_sliding_delay(ref, seg, window_size, step, fs=1.0,
                         plot_result=False, ax=None, save_fig=False, fig_path="window_sliding_delay.png"):
    """
    Slide a window across the signals and compute the delay for each window.
    Optionally plot the window center times vs. delays on a fixed axes and save the figure.

    Parameters:
        ref (np.ndarray): The full reference signal.
        seg (np.ndarray): The full segment signal.
        window_size (int): Number of samples per window.
        step (int): Step size (in samples) between consecutive windows.
        fs (float): Sampling frequency in Hz.
        plot_result (bool): If True, plot the results.
        ax (matplotlib.axes.Axes or None): A fixed axes to plot on. If None, a new figure and axes are created.
        save_fig (bool): If True, save the plot to a file.
        fig_path (str): File path for saving the figure.
        
    Returns:
        times (np.ndarray): Array of central sample indices for each window.
        dt_array (np.ndarray): Array of delays (in seconds) computed for each window.
    """
    dt_list = []
    time_list = []
    N = len(ref)
    
    for start in range(0, N - window_size + 1, step):
        ref_win = ref[start:start+window_size]
        seg_win = seg[start:start+window_size]
        corr, dt = crosscorr_delay(ref_win, seg_win, fs)
        dt_list.append(dt)
        center = start + window_size // 2
        time_list.append(center)
        
        times = np.array(time_list)
    dt_array = np.array(dt_list)
    
    # Optional plotting.
    if plot_result:
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(times / fs, dt_array, marker='o', linestyle='-')
        ax.set_xlabel("Window Center Time (s)")
        ax.set_ylabel("Estimated Delay (s)")
        ax.set_title("Windowed Delay Estimation via Cross-Correlation")
        ax.grid(True)
        if save_fig:
            plt.savefig(fig_path)
        plt.show()
    
    return times, dt_array


def compute_cwi(reference, segments, window_size, step, fs=1.0, axes=None):
    """
    For a given reference waveform and a list of segmented waveforms, compute the 
    windowed time shifts (coda wave interferometry) for each segment and derive a single
    estimation of dt/t from the slope of the time shifts versus time.
    
    Parameters:
        reference (np.ndarray): 1D baseline coda waveform.
        segments (list of np.ndarray): List of 1D waveforms (each a segment to compare).
        window_size (int): Number of samples per window.
        step (int): Step size for sliding window (in samples).
        fs (float): Sampling frequency in Hz.
        
    Returns:
        cwi_results (list): List of dictionaries, one per segment, each containing:
            - 'segment_id': Identifier for the segment.
            - 'window_times': Array of window central times (in seconds).
            - 'time_shifts': Array of computed time shifts (in seconds) for each window.
            - 'dt_over_t_est': Single estimation of dt/t derived from the slope of time_shifts vs. time.
            - 'slope': The fitted slope (which is dt/t).
            - 'intercept': The intercept from the linear regression.
    """
    
    cwi_results = []
    for i, seg in enumerate(segments):
        # Compute windowed time shifts for the current segment.
        times, dt_array = window_sliding_delay(reference, seg, window_size, step)
        # Convert window central sample indices to seconds.
        times_sec = times / fs
        
        if axes:
            axes.plot(times_sec, dt_array)
        # Fit a line to dt_array vs. times_sec using a linear regression.
        # The linear model is: dt = slope * t + intercept.
        slope, intercept = np.polyfit(times_sec, dt_array, 1)
        
        # The slope of the fit provides a single estimation of dt/t.
        dt_over_t_est = slope
        
        result = {
            'segment_id': i,
            'dt_over_t_est': dt_over_t_est,
            'slope': slope,
            'intercept': intercept
        }
        cwi_results.append(result)
    return cwi_results