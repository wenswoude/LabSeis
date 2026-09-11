import numpy as np
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