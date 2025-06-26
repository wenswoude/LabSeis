#
#For general purpose IO and utilitis 
#

import mmap
import pandas as pd
import numpy as np
import h5py as h5py
import labseis.tpc5 as tpc5

import os
import glob

def csv_mmap_row_offsets(filename):
    row_offsets = [0]
    with open(filename, "r") as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        while True:
            pos = mmapped_file.tell()  # current position at start of the line
            line = mmapped_file.readline()
            if not line:
                break
            row_offsets.append(mmapped_file.tell())
    return row_offsets

def csv_mmap_read(filename, row_offsets, start_row, num_rows):
    with open(filename, "r") as f:
        mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Seek to the start of the desired row
        mmapped_file.seek(row_offsets[start_row])
        
        selected_rows = []
        for _ in range(num_rows):
            line = mmapped_file.readline()
            if not line:
                break
            # Process line: decode and convert values to int
            int_values = list(map(int, line.decode().strip().split(',')))
            selected_rows.append(int_values)

    return np.array(selected_rows).T


def read_tpc5(f, block):
    tmp = tpc5.getVoltageData(f,1,block=block)
    t = 1000*np.arange(len(tmp))/tpc5.getSampleRate(f,1,block) # time axis in ms
    timestamp = tpc5.getTriggerTime(f,2,block)

    AE = np.empty([17,len(tmp)])
    AE[0] = tmp

    for i in range(1,17):
        AE[i] = tpc5.getVoltageData(f,i+1,block=block)
    
    return AE, t, timestamp