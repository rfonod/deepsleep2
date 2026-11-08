import os
import numpy as np
import h5py

def import_signals(file_name): 
    import scipy.io
    signals = scipy.io.loadmat(file_name)['val']
    return signals

def import_arousals(file_name):
    with h5py.File(file_name, 'r') as f:
        arousals = np.array(f['data']['arousals']).transpose()
    return np.int8(arousals)

def folders_to_records_txt(data_path):
    from os import walk
    from os.path import join

    subsubfolders = []

    _, folders, _ = next(walk(data_path))
    
    for folder in sorted(folders):
        # skip hidden folders
        if folder[0] == '.':
            continue
        _, subsubfolders_, _ = next(walk(join(data_path, folder)))

        subfolders = []
        for subsubfolder in sorted(subsubfolders_):
            # skip hidden folders in the subfolders
            if subsubfolder[0] == '.':
                continue
            subfolders.append(join(subsubfolder, subsubfolder))
            subsubfolders.append(join(folder, subsubfolder, subsubfolder))

        with open(join(data_path, folder, 'RECORDS.txt'),'w') as f:
            f.write('\n'.join(subfolders))

    with open(join(data_path,'RECORDS.txt'),'w') as f:
        f.write('\n'.join(subsubfolders))

def preprocess(data_path = './data/', z_normalization = False, override = False):
    """Pad every record listed in `<data_path>/RECORDS.txt` to 2^23 samples (optionally
    z-normalizing it) and cache it as `.h5py`. Arousal labels are converted as well
    whenever the record has a `-arousal.mat` file (the PhysioNet `test` folder has none)."""
    max_len = 2**23
    file_add = '_norm' if z_normalization else ''
    
    with open(os.path.join(data_path, 'RECORDS.txt'), 'r') as f:
        all_ids = f.read().splitlines()

        for id_idx, id_name in enumerate(all_ids):
            
            file_path = os.path.join(data_path, id_name)    
            
            if override or not os.path.isfile(file_path + file_add + '.h5py'):
        
                print('|', format(id_idx + 1, '04d'), '/', format(len(all_ids), '04d'), \
                      '-', id_name[-9::], '(*) ', sep = '', end = ' ')                

                # import the signals and arousals (if exist)
                signals = import_signals(file_path + '.mat')
                has_arousals = os.path.isfile(file_path + '-arousal.mat')
                if has_arousals:
                    arousals = import_arousals(file_path + '-arousal.mat')

                # normalize the signal, if applicable
                if z_normalization:
                    signals_mean = np.mean(signals, axis = 1, keepdims = True)
                    signals_std = np.std(signals, axis = 1, keepdims = True, ddof = 1)
                    signals = np.float16(np.divide((signals - signals_mean), signals_std, \
                                                   out=np.zeros(signals.shape, dtype=float), \
                                                   where = signals_std != 0))

                # perform padding 
                padd = max_len - signals.shape[1]
                if padd > 0:
                    signals = np.pad(signals, ((0,0),(padd//2 + padd%2, padd//2)))            
                    if has_arousals:
                        arousals = np.pad(arousals, ((0,0),(padd//2 + padd%2, padd//2)), \
                                          constant_values = -1)

                # save to h5 files
                with h5py.File(file_path + file_add + '.h5py', 'w') as f:
                    f.create_dataset('data', data = signals)
                    
                if has_arousals:
                    with h5py.File(file_path + '-arousal.h5py', 'w') as f:
                        f.create_dataset('data', data = arousals)
                        
def get_record(file_path, channels, file_format = 1):
    arousal_file = file_path + ('-arousal.mat' if file_format in {0, 'raw'} else '-arousal.h5py')
    if not os.path.isfile(arousal_file):
        raise FileNotFoundError('Arousal labels not found: ' + arousal_file + '. Only '
                                'labeled records (the PhysioNet training folder) can be '
                                'used for training and evaluation.')

    file_add = '_norm' if file_format in {2, 'processed_norm'} else ''
    if file_format in {1, 2, 'processed', 'processed_norm'}:
        with h5py.File(file_path + file_add + '.h5py', 'r') as f:
            recording = f['data'][channels]
        with h5py.File(file_path + '-arousal.h5py', 'r') as f:
            arousal = f['data'][:]          
    elif file_format == 0 or file_format == 'raw':
        recording = import_signals(file_path + '.mat')[channels,:]
        arousal = import_arousals(file_path + '-arousal.mat')
    else:
        raise ValueError('Unknown file_format: ' + repr(file_format))

    return recording, arousal
def latest_checkpoint(model_path):
    """Return the path of the highest-numbered `my_checkpoint_<epoch>.pth.tar` in
    `model_path`, or None if the folder holds no checkpoint."""
    import re

    pattern = re.compile(r'^my_checkpoint_(\d+)\.pth\.tar$')
    epochs = [int(m.group(1)) for m in map(pattern.match, os.listdir(model_path)) if m]
    if not epochs:
        return None
    return os.path.join(model_path, 'my_checkpoint_' + str(max(epochs)) + '.pth.tar')
