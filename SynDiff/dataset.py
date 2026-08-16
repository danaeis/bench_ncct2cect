import torch.utils.data
import numpy as np, h5py
import random


def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2'):

    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
    data_fs_s1=LoadDataSet(target_file)
    
    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
    data_fs_s2=LoadDataSet(target_file)

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))  
    return dataset 



#Dataset loading from load_dir and converintg to 256x256
def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True, chunk = 512):
    """Load one contrast of a data_<phase>_<contrast>.mat into a float32 array.

    MEMORY. This function is why a SynDiff run can take a shared box down. The
    dataset is held in RAM in full (upstream feeds it to a TensorDataset), which
    for our VinDr split is 25,251 x 256 x 256 x 4B = 6.6 GB per contrast — that
    part is upstream's design and is unavoidable here. What was avoidable was
    everything on top of it. The original:

        if np.array(f[variable]).ndim == 3:      # materialises all 6.6 GB...
            data = np.expand_dims(np.transpose(np.array(f[variable]), ...))
                                              # ...throws it away, reads it AGAIN
        data = data.astype(np.float32)           # full copy #3
        data = np.pad(data, ...)                 # full copy #4 (even when pad=0)
        data = (data - 0.5) / 0.5                # full copy #5

    read the whole dataset twice and then made three more full-size copies, so
    peak was ~4x the steady-state footprint — over 25 GB for the training split,
    per contrast, and both contrasts load back to back.

    Now: `.ndim` is read from the HDF5 header, the data is streamed into one
    preallocated output buffer, the zero-width pad is skipped, and the
    normalisation is in place. Peak is the array plus one `chunk`-sized slab.
    Output values are identical.
    """
    with h5py.File(load_dir, 'r') as f:
        ds = f[variable]
        if ds.ndim == 3:
            # (N, W, H) -> (N, 1, H, W), transposed per MATLAB column-major
            # convention. Streamed: never holds a second full-size array.
            n, w, h = ds.shape
            data = np.empty((n, 1, h, w), dtype=np.float32)
            for i in range(0, n, chunk):
                data[i:i + chunk, 0] = np.transpose(ds[i:i + chunk], (0, 2, 1))
        else:
            data = np.ascontiguousarray(
                np.transpose(ds[()], (1, 0, 3, 2)), dtype=np.float32)

    if padding:
        pad_x=int((256-data.shape[2])/2)
        pad_y=int((256-data.shape[3])/2)
        print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
        # np.pad always copies; at our size the pad is 0 and the copy is pure waste.
        if pad_x or pad_y:
            data=np.pad(data,((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))
    if Norm:
        # in place: (data - 0.5) / 0.5 == 2*data - 1
        data -= 0.5
        data *= 2.0
    return data
