import xarray as xr
import numpy as np

# def load_and_preprocess_data(data_path, channel_list, SamplingRate):
#     """
#     Loads seismic data from HDF5, performs signal processing, and prepares Xarray datasets.

#     Parameters
#     ----------
#     data_path : str
#         Path to the HDF5 data file.
#     channel_list : list of str
#         List of channel names to process.
#     SamplingRate : float
#         The sampling rate of the data.

#     Returns
#     -------
#     data_xr : xr.Dataset
#         The final Xarray Dataset containing the processed waveforms.
#     """
#     import h5py as h5py
#     import tpc5 # Assuming tpc5 is available in the environment
    
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"Data file not found at: {data_path}")

#     f = h5py.File(data_path, 'r')
    
#     # Load metadata
#     RecTimeString = tpc5.getStartTime(f, channel=1, block=1)
#     RecTimeList   = RecTimeString.split('T',1)
#     RecDate       = RecTimeList[0]
#     TimeList      = RecTimeList[1].split('.',1)
#     RecTime       = TimeList[0]
#     TriggerSample = tpc5.getTriggerSample(f, channel=1, block=1)
#     SamplingRate  = tpc5.getSampleRate(f, channel=1, block=1)

#     # Load AE data
#     channel_list = []
#     i = 1
#     while True:
#         try:
#             channel = tpc5.getChannelName(f, channel=i)
#             channel_list.append(channel)
#         except:
#             break
    
#     num_blocks = tpc5.getNumBlocks(f, channel=1)
#     AE = np.stack([tpc5.getVloltageDataLimited(f, channel=c, block=b) for b in range(1, num_blocks + 1) for c in range(len(channel_list)])
    
#     # Stack and process
#     AE_stacked = np.mean(AE, axis=0)
    
#     from scipy.signal import bandpass
#     import scipy as sp
#     ns = len(AE_stacked[0])
#     tukeytaper = sp.signal.windows.tukey(ns, alpha=0.1, sym=True)
#     AE_stacked = AE_stacked * tukeytaper
#     AE_stacked_filtered = bandpass(AE_stacked, 1/SamplingRate, 50000.0, 100000.0)
    
#     return xr.Dataset({
#         "waveform": ( ("sensor", "time"), AE_stacked_filtered )
#     }, coords={
#         "sensor": np.arange(len(channel_list)),
#         "time": np.arange(len(AE_stacked_filtered.shape[1])) / SamplingRate
#     })

# ---------------------------------------------------
# 7️⃣  3‑D travel‑time (use full sensor coordinates)
# ---------------------------------------------------
def homogeneous_travel_time_3d(fault_pts, rec_coords, velocity):
    """Compute travel‑time matrix using full 3‑D coordinates.

    Parameters
    ----------
    fault_pts : np.ndarray (N, 3)
        Source locations (x, y, z) in metres.
    rec_coords : np.ndarray (M, 3)
        Receiver coordinates (x, y, z) in metres.
    velocity : float
        Homogeneous wave speed (m/s).
    """
    diff = rec_coords[:, np.newaxis, :] - fault_pts[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)  # (M, N)
    return distances / velocity

# Update migration to use 3‑D coordinates if available
def migrate_shift_stack_3d(data, rec_coords, fault_pts, velocity, fs, ref_channel=0):
    """Shift‑and‑stack migration using full 3‑D geometry.
    Returns stacked_best, stacked_per_source, shifts.
    """
    tt = homogeneous_travel_time_3d(fault_pts, rec_coords, velocity)
    ref_tt = tt[ref_channel, :]
    shifts = np.rint((tt - ref_tt) * fs).astype(int)
    M, L = data.shape
    N = fault_pts.shape[0]
    stacked_per_source = np.zeros((N, L))
    for i in range(N):
        aligned = np.empty_like(data)
        for ch in range(M):
            shift = shifts[ch, i]
            aligned[ch] = np.roll(data[ch], -shift)
            if shift > 0:
                aligned[ch, -shift:] = 0
            elif shift < 0:
                aligned[ch, :-shift] = 0
        stacked_per_source[i] = np.mean(aligned, axis=0)
    energy = np.sum(stacked_per_source ** 2, axis=1)
    best_idx = np.argmax(energy)
    stacked_best = stacked_per_source[best_idx]
    return stacked_best, stacked_per_source, shifts