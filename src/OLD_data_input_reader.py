import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def read_segmented_images(data_input, microscope='sted', replace_nan_with=None):

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
            keys = list(npfile.keys())
            
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
