
from scipy.io import loadmat

def loadAEmat(filename):
    tpd = loadmat(filename)
    tpd = tpd['tpd']
    meta = {
        'name': tpd['Name'][0],
        'date_time': tpd['DateTime'][0],
        'range_min': tpd['RangeMin'][0, 0],
        'range_max': tpd['RangeMax'][0, 0],
        'sample_frequency': tpd['SampleFrequency'][0, 0],
        'pre_sample_count': tpd['PreSampleCount'][0, 0],
        'start_value': tpd['StartValue'][0, 0],
        'unit': tpd['Unit'][0]
    }
    data_matrix = tpd['Data'][0][0]
    return meta, data_matrix



def loadAEcsv(filename):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(filename, skiprows=8, header=0, sep=';',names=['time', 'ch1', 'ch2'], usecols=[0, 1, 2])
    # Extract data matrix similar to the 'data_matrix' structure
    data_matrix = np.array([df['ch1'].values, df['ch2'].values], dtype=np.float32)
    meta = {
        'name': np.array([['ch1', 'ch2']], dtype=object),
        'sample_frequency': np.array([[1 / (df.time[1] - df.time[0])]]),
        'unit': np.array([[['V'], ['V']]], dtype=object)
    }
    return meta, data_matrix
