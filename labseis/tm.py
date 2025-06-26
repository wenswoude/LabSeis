import glob
import re
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def normalized_cross_correlation(signal, template):
    m = len(template)
    n = len(signal)
    
    # Normalize the template
    template_norm = (template - np.mean(template)) / np.std(template)
    
    # Compute the raw cross-correlation (valid mode)
    raw_corr = sp.signal.correlate(signal, template_norm, mode='valid')
    
    # Compute moving average and moving std for the signal
    window = np.ones(m)
    mean_signal = sp.signal.fftconvolve(signal, window, mode='valid') / m
    mean_sq_signal = sp.signal.fftconvolve(signal**2, window, mode='valid') / m
    std_signal = np.sqrt(mean_sq_signal - mean_signal**2)
    
    # Avoid division by zero
    std_signal[std_signal == 0] = 1e-10
    
    # Normalize the raw correlation by the local standard deviation
    norm_corr = raw_corr / (std_signal * m)
    lags = np.arange(n - m + 1)
    
    return norm_corr, lags


def template_matching(signal, template, threshold=0.8, min_distance=None, plot=True):
    """
    Detects segments in 'signal' that match the 'template' using cross-correlation.

    Parameters:
    - signal (1D array): The long time-series data (length n)
    - template (1D array): The master template (length m)
    - threshold (float): Normalized correlation threshold for detecting a match (0 to 1)
    - plot (bool): If True, plots the cross-correlation for visual inspection.

    Returns:
    - match_indices (list): Start indices where the template is detected
    - match_segments (list): Extracted segments matching the template
    """

    n, m = len(signal), len(template)
    if m > n:
        raise ValueError("Template longer than signal")
    if min_distance is None:
        min_distance = max(1, m // 2)


    # Compute cross-correlation
    # correlation = sp.signal.correlate(signal, template, mode='valid', method='fft')
    ncc,lags = normalized_cross_correlation(signal, template)

    # # Normalize the correlation to get similarity values between -1 and 1
    # norm_factor = np.sqrt(np.sum(template**2) * sp.signal.correlate(signal**2, np.ones_like(template), mode='valid'))
    # normalized_corr = correlation / norm_factor  # Element-wise division

    # Find peaks where correlation exceeds the threshold
    peak_indices, _ = sp.signal.find_peaks(ncc, height=threshold)

    # Non-maximum suppression
    peaks_sorted = sorted(peak_indices, key=lambda idx: ncc[idx], reverse=True)

    suppressed_indices = []
    used = np.zeros(len(ncc), dtype=bool)
    
    for peak in peaks_sorted:
        if not used[peak]:
            suppressed_indices.append(peak)
            start = max(0, peak - min_distance)
            end = min(len(ncc), peak + min_distance + 1)
            used[start:end] = True
            
    suppressed_indices.sort()  # Return in chronological order

    # Extract matching segments
    match_segments = [signal[idx:idx+m] for idx in suppressed_indices]


    # Plot the cross-correlation and detected peaks
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Time series plot
        ax1.plot(signal, label='Signal')
        for p in suppressed_indices:
            ax1.axvline(p, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Signal with Detected Segments')
        ax1.legend()
        
        # NCC plot
        ax2.plot(ncc, label='NCC Score')
        ax2.scatter(suppressed_indices, ncc[suppressed_indices], c='r', label='Selected Peaks')
        ax2.axhline(threshold, color='k', linestyle='--', label='Threshold')
        ax2.set_title('Normalized Cross-Correlation')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    return match_segments, np.array(suppressed_indices), ncc[suppressed_indices]



def load_segments_with_indices(file_pattern="segment_*.csv"):
    """
    Load multiple CSV segment files, extract their numeric indices from filenames,
    and concatenate the signal values into one 1D NumPy array.
    
    Parameters:
        file_pattern (str): Pattern to match the segment CSV files.
        
    Returns:
        concatenated_values (np.ndarray): 1D array of concatenated signal values.
        indices_array (np.ndarray): 1D array of the numeric indices extracted from filenames.
    """
    # Get a sorted list of files matching the pattern
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        raise FileNotFoundError("No files found with pattern: " + file_pattern)
    
    segments = []     # List to store the signal arrays from each file
    indices = []      # List to store the numeric indices extracted from filenames
    
    # Define a regular expression to extract the numeric index.
    pattern = re.compile(r'segment_(\d+)\.csv')
    
    for filename in file_list:
        # Extract the index from the filename
        match = pattern.search(filename)
        if match:
            index_val = int(match.group(1))
            indices.append(index_val)
        else:
            # If the filename doesn't match the expected pattern, skip it.
            continue
        
        # Load the CSV file.
        # Assumes the CSV file has a header and that the signal values are in the second column.
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        # Check if data was loaded correctly
        if data.ndim == 1:
            # If there's only one row, ensure data remains 2D for consistency.
            data = data[np.newaxis, :]
        # print(data.shape)
        # Extract the signal values (second column)
        segments.append(data)
    
    # Concatenate all segment arrays into one 1D array.
    segments = np.array(segments)
    indices_array = np.array(indices)

    # indices_array should be monolically asending.
    idx_sort = np.argsort(indices_array)
    
    return segments[idx_sort], indices_array[idx_sort]