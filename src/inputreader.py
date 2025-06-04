import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def read_segmented_images(data_input, microscope='sted', replace_nan_with=None):
    """
    This function reads in the segmented images.
    """

    keys_sted = ['bounding_box', 'mask_2d', 'mask_3d', 'params_keys', 'params', 'values']
    keys_airyscan = ['bounding_boxes'] # and possibly '00' and '01'

    masks = []
    bounding_boxes = []
    labels_str = []
    if microscope.lower() == 'sted':
        npzfiles = list(Path(data_input).rglob('*.npz'))
        idx = np.array([[int(f.stem.split('_')[-2]), int(f.stem.split('_')[-1])] for f in npzfiles])
        idx = np.lexsort((idx[:, 1], idx[:, 0]))
        npzfiles = [npzfiles[i] for i in idx]

        for file in tqdm(npzfiles):
            npfile = np.load(file)
            # keys = list(npfile.keys())
            
            # check if all keys are present
            for keys_check in keys_sted:
                assert keys_check in npfile.keys()

            if replace_nan_with is not None:
                img = npfile['values']
                img[np.isnan(img)] = replace_nan_with
                masks.append(img.copy())
            else:
                masks.append(npfile['values'].copy())

            bounding_boxes.append(npfile['bounding_box'])
            if file.stem.split('_')[1] == 'ES':
                labels_str.append('ES')
            elif file.stem.split('_')[1] == 'NS':
                labels_str.append("NS")
            else:
                raise ValueError('Invalid label.', file)

    elif microscope.lower() == 'airyscan':

        npzfiles = list(Path(data_input).rglob('*.npz'))
        idx = [[int(file.stem[file.stem.index('idfile'):].split('_')[0].split('-')[1]),
                int(file.stem[file.stem.index('idscene'):].split('_')[0].split('-')[1])]
                for file in npzfiles]
        idx = np.array(idx)

        idx = np.lexsort([idx[:, 1], idx[:, 0]])
        npzfiles = [npzfiles[i] for i in idx]

        # read in the segmented files
        
        npzfiles_list = []
        for file in tqdm(npzfiles):
            npfile = np.load(file)

            if len(npfile.keys()) == 1:
                continue
            
            typelabel = file.stem.split('_')[0]
            if len(npfile.keys()) == 2:
                assert np.shape(npfile['bounding_boxes'])[0] == 1

                masks.append(npfile['00'])
                bounding_boxes.append(npfile['bounding_boxes'])
                labels_str.append(file.stem.split('_')[0])
                npzfiles_list.append(file)
            else:
                assert len(npfile.keys()) == 3
                assert np.shape(npfile['bounding_boxes'])[0] == 2

                # first mask
                masks.append(npfile['00'])
                bounding_boxes.append(npfile['bounding_boxes'][0])
                labels_str.append(file.stem.split('_')[0])
                npzfiles_list.append(file)
                # second mask        
                masks.append(npfile['01'])
                bounding_boxes.append(npfile['bounding_boxes'][1])
                labels_str.append(file.stem.split('_')[0])
                npzfiles_list.append(file)
        npzfiles = npzfiles_list
    labels = np.array([0 if lbl == 'ES' else 1 for lbl in labels_str])
    return masks, bounding_boxes, labels_str, labels, npzfiles


def persistence_writer_ensure_filesize(filepath,
        maxfilesize=95, lowerlimit=90,
        split_names='splitfiles',
        pers2d3d=['pers-2', 'pers-3']):
    """
    This function ensures that the file size of the persistence files is below a certain threshold.

    Parameters
    ----------
    filepath : Path
        The path to the file that should be split.
    maxfilesize : float, optional
        The maximum file size in MB.
        The default is 95.
    lowerlimit : float, optional
        To ensure that the file size is below, we pick a smaller value than maxfilesize.
        The default is 90.
    split_names : str, optional
        The name of the split files. The default is 'splitfiles'.   
    pers2d3d : list, optional
        The names of the 2d and 3d persistence files.
        The default is ['pers-2', 'pers-3'].
    """
    temporaryfilesToDelete = []

    assert lowerlimit <= maxfilesize
    filepath = Path(filepath)
    assert filepath.is_file()

    npfile = np.load(filepath)
    if (filepath.is_file() and ~filepath.name.startswith(split_names)
            and filepath.stat().st_size / (1024*1014) >= maxfilesize):
        # save the persistenc computed on 2d and 3d separately
        newfile2d = [key for key in npfile.keys() if pers2d3d[0] in key]
        newfile3d = [key for key in npfile.keys() if pers2d3d[1] in key]
        newfile_names = [Path(filepath.parents[0], f'{split_names}_{perstype}__{filepath.name}')
                         for perstype in pers2d3d]
        np.savez_compressed(newfile_names[0], **{key: npfile[key] for key in newfile2d})
        np.savez_compressed(newfile_names[1], **{key: npfile[key] for key in newfile3d})

        for newfile in newfile_names:
            filesize = newfile.stat().st_size / (1024*1014)
            if filesize > lowerlimit:
                temporaryfilesToDelete.append(newfile)

                filekeys = list(np.load(newfile).keys())

                splits = [1 / np.ceil(filesize / lowerlimit)
                          for _ in range(int(np.ceil(filesize / lowerlimit)))]
                splits[-1] = 1 - np.sum(splits[:-1])
                assert np.sum(splits) == 1

                numkeys_split = [int(np.ceil(len(filekeys) * x)) for x in splits]
                numkeys_split[-1] = len(filekeys) - np.sum(numkeys_split[:-1])
                numkeys_split = np.array([0] + numkeys_split).astype(np.int32)
                numkeys_split = np.cumsum(numkeys_split)
                assert numkeys_split[-1] == len(filekeys)

                keys = [filekeys[numkeys_split[i]:numkeys_split[i+1]]
                        for i in range(len(numkeys_split) - 1)]
                key_id = 0
                for i, key in enumerate(keys):
                    newfile_name = Path(newfile.parents[0],
                        f'{newfile.name.split("__")[0]}_split-{key_id:03d}__{newfile.name.split("__")[1]}')
                    np.savez_compressed(newfile_name,
                        **{keyi: npfile[keyi] for keyi in key})

                    assert newfile_name.is_file()
                    if newfile_name.stat().st_size / (1024*1014) >= maxfilesize:

                        splitkey = [key[:len(key)//2], key[len(key)//2:]]

                        # overwrite the first one
                        newfile_name = Path(newfile.parents[0],
                            f'{newfile.name.split("__")[0]}_split-{key_id:03d}__{newfile.name.split("__")[1]}')
                        np.savez_compressed(newfile_name,
                            **{keyi: npfile[keyi] for keyi in splitkey[0]})
                        assert newfile_name.stat().st_size / (1024*1014) <= maxfilesize
                        print(newfile_name, newfile_name.stat().st_size / (1024*1014))

                        # now do the next one
                        key_id += 1
                        newfile_name = Path(newfile.parents[0],
                            f'{newfile.name.split("__")[0]}_split-{key_id:03d}__{newfile.name.split("__")[1]}')
                        np.savez_compressed(newfile_name,
                            **{keyi: npfile[keyi]for keyi in splitkey[1]})
                        assert newfile_name.stat().st_size / (1024*1014) <= maxfilesize
                        print(newfile_name, newfile_name.stat().st_size / (1024*1014))

                    else:
                        key_id += 1
                        print(newfile_name, newfile_name.stat().st_size / (1024*1014))

    for file in temporaryfilesToDelete:
        if file.is_file():
            file.unlink()
    
    return None


def read_persistence_files(data_input,
        preprocessing='clip_minmax_gaussian_2c_minmax',
        return_keys=False,
        include_subdirs=False):
    """
    This function reads in the persistence files.
    
    Parameters
    ----------
    data_input : Path or str
        The path to the persistence files.
    preprocessing : str, optional
        The preprocessing type. The default is 'clip_minmax_gaussian_2c_minmax'.
    """

    data_input = Path(data_input)
    assert data_input.is_dir()

    if 'mask0' in preprocessing:
        if include_subdirs:
            # include subdirectories
            files = sorted([x for x in data_input.rglob('*.npz')
                if 'persistence' in x.name
                and preprocessing in x.name])
        else:
            files = sorted([x for x in data_input.glob('*.npz')
                if 'persistence' in x.name
                and preprocessing in x.name
                and 'mask0' in x.name])
    else:
        if include_subdirs:
            # include subdirectories
            files = sorted([x for x in data_input.rglob('*.npz')
                if 'persistence' in x.name
                and preprocessing in x.name])
        else:
            files = sorted([x for x in data_input.glob('*.npz')
                if 'persistence' in x.name
                and preprocessing in x.name
                and 'mask0' not in x.name])

    if 'gauss' not in preprocessing:
        # only consider files with 'gauss' in the name
        files = sorted([x for x in files if 'gauss' not in x.name])
    if 'minmax' not in preprocessing:
        # only consider files with 'minmax' in the name
        files = sorted([x for x in files if 'minmax' not in x.name])

    # only consider splitfiles
    if np.any(['splitfiles' in x.name for x in files]):
        files = sorted([x for x in files if 'splitfiles' in x.name])

    if len(files) == 0:
        raise ValueError('No files found.')

    # get and sort the keys
    keys = [[file, x] for file in files for x in np.load(file).keys()
            if x != 'labels']
    vals = np.array([[int(x[1].split('_')[0].split('-')[1]),
                    int(x[1].split('_')[1].split('-')[1]),
                    int(x[1].split('_')[2].split('-')[1])]
                    for x in keys])
    # check for consistency of the keys
    assert np.all([x[1].split('_')[0].startswith('pers-')
        and x[1].split('_')[1].startswith('dim-')
        and x[1].split('_')[2].startswith('i-')
        for x in keys])
    for dimtype in np.unique(vals[:, 0]):
        dimvals = np.unique(vals[vals[:, 0] == dimtype, 1])
        assert len(dimvals) == dimtype
        for dim in dimvals:
            assert np.all(vals[(vals[:, 0] == dimtype) \
                & (vals[:, 1] == 0), 2] == vals[(vals[:, 0] == dimtype)\
                & (vals[:, 1] == dim), 2])

    # now sort the keys according to the first, second, and third number
    vals = np.array([[int(x[1].split('_')[0].split('-')[1]),
                    int(x[1].split('_')[1].split('-')[1]),
                    int(x[1].split('_')[2].split('-')[1])]
                    for x in keys])
    idx = np.lexsort([vals[:, 2], vals[:, 1], vals[:, 0]])
    keys = [keys[i] for i in idx]
    vals = vals[idx, :]

    pers_all = {2: [[], []], 3: [[], [], []]}
    for key, val in zip(keys, vals):
        pers_all[val[0]][val[1]].append(np.load(key[0])[key[1]])

    if return_keys:
        return pers_all, np.array(keys)

    return pers_all
