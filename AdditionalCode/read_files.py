"""
Methods for reading in ciz./lif files in the data and path layouts provided.
"""

# %%
# import packages

import numpy as np
import pandas as pd
from pathlib import Path
import aicsimageio
from tqdm import tqdm

# %%
#


def read_cizlif_files_get_data(filepath, date=None,
                               images=None,
                               metadata=None,
                               file_info=None,
                               exclude_one_slice=True):
    """Read file (in either ciz or lif format) and return it's array,
    Depending on the date value we either get channel 0 or 1 (if there
    is more than one).

    If there is more than one scene also save all scenes as different images.
    We do NOT save slices of images, but discard them.


    return an array in the form z-y-x
    as well as the metadata of pixel_sizes for z,y,x
    """
    if Path(filepath).suffix == '.czi':
        img = aicsimageio.readers.czi_reader.CziReader(filepath)
        dims_strings = img.mapped_dims

    elif Path(filepath).suffix == '.lif':
        img = aicsimageio.readers.lif_reader.LifReader(filepath)
        dims_strings = img.dims._order
    else:
        print('wrong file extension:', Path(filepath).suffix)
        return (None)

    # if images and metadata are not already given
    if images is None:
        images = []
    if metadata is None:
        metadata = []

    # if date is None check if the folder has
    # a readme which contains the line Ctcf-Tmr
    # and contains 
    if date is None:
        readme = list(filepath.parent.rglob('readme.txt'))
        if len(readme) == 1:
            with open(readme[0], "r") as file:
                for line in file:
                    if "Ctcf-Tmr" in line:
                        idx = line.find('C=')
                        date = int(line[idx+2])
                        break
        elif len(readme) > 1:
            raise FileExistsError('There is not one unique file named readme.txt in the folder!')
        else:
            date = 0

        if date is None:
            raise ValueError('Date should have been set!')
    # print('Date =', date)

    # now go through all scenes and save them as different images
    count = 0
    for scene_id in img.scenes:
        try:
            # select the scene
            img.set_scene(scene_id)
            tmp_info = {'id_scene': scene_id,
                        'pixel_size_z': img.physical_pixel_sizes[0],
                        'pixel_size_y': img.physical_pixel_sizes[1],
                        'pixel_size_x': img.physical_pixel_sizes[2]}
            if file_info is not None:
                # join the two dictionaries (overwriting information in the first)
                tmp_info = file_info | tmp_info 

            if 'H' in dims_strings:
                dim_h = dims_strings.index('H')
                assert np.shape(img.data)[dim_h] == 1

            # print(dims_strings, filepath, scene_id)

            dim_t = dims_strings.index('T')
            assert np.shape(img.data)[dim_t] == 1

            dim_c = dims_strings.index('C')
            dim_z = dims_strings.index('Z')
            dim_y = dims_strings.index('Y')
            dim_x = dims_strings.index('X')

            if exclude_one_slice:
                for dim_i in [dim_z, dim_y, dim_x]:
                    if np.shape(img.data)[dim_i] == 1:
                        print(f'only one z_dimension, {scene_id}, {filepath}!')
                        raise ValueError(f'only one z_dimension, {scene_id}, {filepath}!')

            channel = 0
            if np.shape(img.data)[dim_c] != 1:
                if Path(filepath).suffix == '.lif':
                    print('No channel information given for lif')
                    print(filepath)
                    raise ValueError

                # the date 30.12.22 should not have more than one channel!
                if date == 'original' or date == 0:
                    channel = 0
                elif date == '28.12.22' or date == 1:
                    channel = 1
                elif isinstance(date, int):
                    channel = date
                else:
                    print(np.shape(img.data), dim_c, Path(filepath).suffix)
                    print(date, np.shape(img.data)[dim_c])
                    print(f'Wrong dimensions for channel and date, {date}, {scene_id}, {filepath}!')
                    raise ValueError(f'Wrong dimensions for channel and date, {date}, {scene_id}, {filepath}!')

            # bring them in the format Z, Y, X while only picking the appropriate channel
            if Path(filepath).suffix == '.czi':
                img_arr = np.transpose(img.data, (dim_h, dim_t, dim_c, dim_z, dim_y, dim_x))
                img_arr = img_arr[0, 0, channel, :, :, :]
            elif Path(filepath).suffix == '.lif':
                img_arr = np.transpose(img.data, (dim_t, dim_c, dim_z, dim_y, dim_x))
                img_arr = img_arr[0, channel, :, :, :]

            tmp_info['id_scene_count'] = int(count)
            images.append(img_arr)
            metadata.append(tmp_info)

            count += 1

        except ValueError:
            # it can occur that scenes are in the file that do not give numpy arrays
            continue

    return (images, metadata)


def read_cizlif_file(filepath,
                     scene_id,
                     date=None,
                     file_info=None,
                     exclude_one_slice=True):
    """Read file (in either ciz or lif format) and return it's array,
    Depending on the date value we either get channel 0 or 1 (if there
    is more than one).

    If there is more than one scene also save all scenes as different images.
    We do NOT save single slices of images, but discard them.

    return an array in the form z-y-x
    as well as the metadata of pixel_sizes for z,y,x
    """
    if Path(filepath).suffix == '.czi':
        img = aicsimageio.readers.czi_reader.CziReader(filepath)
        dims_strings = img.mapped_dims

    elif Path(filepath).suffix == '.lif':
        img = aicsimageio.readers.lif_reader.LifReader(filepath)
        dims_strings = img.dims._order
    else:
        print('wrong file extension:', Path(filepath).suffix)
        return (None)

    # if date is None check if the folder has
    # a readme which contains the line Ctcf-Tmr
    # and contains
    if date is None:
        readme = list(filepath.parent.rglob('readme.txt'))
        if len(readme) == 1:
            with open(readme[0], "r") as file:
                for line in file:
                    if "Ctcf-Tmr" in line:
                        idx = line.find('C=')
                        date = int(line[idx+2])
                        break
        elif len(readme) > 1:
            raise FileExistsError('There is more than one file named readme.txt in the folder!')
        else:
            date = 0

        if date is None:
            raise ValueError('Date should have been set!')
    # print('Date =', date)

    # now go through all scenes and save them as different images
    try:
        # select the scene
        img.set_scene(scene_id)
        tmp_info = {'id_scene': scene_id,
                    'pixel_size_z': img.physical_pixel_sizes[0],
                    'pixel_size_y': img.physical_pixel_sizes[1],
                    'pixel_size_x': img.physical_pixel_sizes[2]}
        if file_info is not None:
            # join the two dictionaries (overwriting information in the first)
            tmp_info = file_info | tmp_info 

        if 'H' in dims_strings:
            dim_h = dims_strings.index('H')
            assert np.shape(img.data)[dim_h] == 1

        # print(dims_strings, filepath, scene_id)

        dim_t = dims_strings.index('T')
        assert np.shape(img.data)[dim_t] == 1

        dim_c = dims_strings.index('C')
        dim_z = dims_strings.index('Z')
        dim_y = dims_strings.index('Y')
        dim_x = dims_strings.index('X')

        if exclude_one_slice:
            for dim_i in [dim_z, dim_y, dim_x]:
                if np.shape(img.data)[dim_i] == 1:
                    print(f'only one z_dimension, {scene_id}, {filepath}!')
                    raise ValueError(f'only one z_dimension, {scene_id}, {filepath}!')

        channel = 0
        if np.shape(img.data)[dim_c] != 1:
            if Path(filepath).suffix == '.lif':
                print('No channel information given for lif')
                print(filepath)
                raise ValueError

            # the date 30.12.22 should not have more than one channel!
            if date == 'original' or date == 0:
                channel = 0
            elif date == '28.12.22' or date == 1:
                channel = 1
            elif isinstance(date, int):
                channel = date
            else:
                print(np.shape(img.data), dim_c, Path(filepath).suffix)
                print(date, np.shape(img.data)[dim_c])
                print(f'Wrong dimensions for channel and date, {date}, {scene_id}, {filepath}!')
                raise ValueError(f'Wrong dimensions for channel and date, {date}, {scene_id}, {filepath}!')

        # bring them in the format Z, Y, X while only picking the appropriate channel
        if Path(filepath).suffix == '.czi':
            img_arr = np.transpose(img.data, (dim_h, dim_t, dim_c, dim_z, dim_y, dim_x))
            img_arr = img_arr[0, 0, channel, :, :, :]
        elif Path(filepath).suffix == '.lif':
            img_arr = np.transpose(img.data, (dim_t, dim_c, dim_z, dim_y, dim_x))
            img_arr = img_arr[0, channel, :, :, :]

    except ValueError:
        # it can occur that scenes are in the file that do not give numpy arrays
        pass

    return (img_arr, tmp_info)


# %%

def from_files_to_df(files):
    data = []
    counter = 0
    for file in files:
        if file.parts[-3] == 'different_resolutions':
            tmp = {'filepath': file.as_posix(),
                   'microscope': 'Airyscan',
                   'filename': file.parts[-1],
                   'filetype': file.parts[-1].split('.')[1],
                   'class': 'Wildtype',
                   'type': 'WT',
                   'cell': 'NS',
                   'comparison': 'resolution',
                   'id': int(file.parts[-1].split('-')[5])
                   }

        else:
            tmp = {'filepath': file.as_posix(),
                   'microscope': file.parts[-2].split('-')[0],
                   'filename': file.parts[-1],
                   'filetype': file.parts[-1].split('.')[1],
                   'comparison': 'all',
                   'id': counter
                   }
            counter += 1

            if 'KO' in file.parts[-3]:
                tmp['class'] = 'KO'
            elif 'Wildtype' in file.parts[-3]:
                tmp['class'] = 'Wildtype'
            else:
                print(f'{file.parts[-3]} does not match class!')

            if 'Wildtype' in file.parts[-3]:
                tmp['type'] = 'WT'
            elif 'Fus' in file.parts[-3]:
                tmp['type'] = 'Fus'
            elif 'Ddx5' in file.parts[-3]:
                tmp['type'] = 'Ddx5'
            # for the new addition of files
            elif 'PA5' in file.parts[-3]:
                tmp['type'] = 'PA5'
            elif 'PA9' in file.parts[-3]:
                tmp['type'] = 'PA9'
            else:
                print(f'{file.parts[-3]} does not match type!')

            if 'NS' in file.parts[-3]:
                tmp['cell'] = 'NS'
            elif 'ES' in file.parts[-3]:
                tmp['cell'] = 'ES'
            # for the new addition of files
            elif 'Pantr' in file.parts[-3]:
                tmp['cell'] = 'Pantr'
            else:
                print(f'{file.parts[-3]} does not match cell type!')

        data.append(tmp)

    df_files = pd.DataFrame(data)
    df_files = df_files[['comparison', 'class', 'type', 'cell',
                         'microscope', 'filename',
                         'filetype', 'filepath']]

    df_files = df_files.sort_values(by=['comparison', 'class', 'type', 'cell'])
    df_files['id_file'] = range(len(df_files))

    # add a column for 'original' data set as well as '28.12.2022' and '30.12.2022'
    df_files['date'] = [x.split('-')[1] if '-' in x else 'original'
                        for x in df_files['microscope'] ]
    # df_files['microscope'] = [x.split('-')[0] if '-' in x else x
    #                           for x in df_files['microscope']]

    df_files = df_files.reset_index(drop=True)
    assert (df_files['id_file'].values == np.arange(len(df_files))).all()

    return (df_files)


def grab_all_subtypes(df_select, exclude_one_slice=True):
    # read in all the images for the scenes which have more than
    # one z-slice
    images = []
    metadata = []
    for fp, id in tqdm(df_select[['filepath', 'id_file']].values):
        # print(fp, id)
        images, metadata = \
            read_cizlif_files_get_data(Path(fp),
                                       images=images,
                                       metadata=metadata,
                                       file_info={'id_file': id},
                                       exclude_one_slice=exclude_one_slice)

    # make the metadata into a dataframe
    metadata = pd.DataFrame(metadata)
    metadata['z'] = np.array([img.shape[0] for img in images],
                             dtype=np.int32)
    metadata['y'] = np.array([img.shape[1] for img in images],
                             dtype=np.int32)
    metadata['x'] = np.array([img.shape[2] for img in images],
                             dtype=np.int32)

    # now save the npz file, use the information from df_files
    # df_merge = df_files.merge(metadata, on='id_file', how='right')
    # if len(df_merge) > len(df_select):
    #     print(' - More scenes than files')

    return (images, metadata)

# %%
#


def read_files_pipeline(data_path, microscope=None,
                        cell=None, cell_type=None,
                        comparison='all',
                        exclude_one_slice=True):
    """Read all files in a certain data path and 
    select the appropiate cell images from the file.

    Parameters
    ----------
    data_path : _type_
        _description_
    microscope : str, optional
        _description_, by default 'Airyscan'
    cell : str, optional
        _description_, by default 'ES'
    cell_type : str, optional
        _description_, by default 'WT'
    comparison : str, optional
        _description_, by default 'all'
    """
    if isinstance(data_path, pd.DataFrame):
        df_files = data_path
    else:
        files = list(data_path.rglob('*.czi'))
        files.extend(list(data_path.rglob('*.lif')))

        # exclude those starting with .
        files = [file for file in files if not file.name.startswith('.')]
        files = [file for file in files if not file.name.startswith('_')]

        df_files = from_files_to_df(files)

    conditions = np.ones(len(df_files), dtype=bool)
    for condition, name in zip([microscope, cell, cell_type, comparison],
                               ['microscope', 'cell', 'type',
                                'comparison']):
        print(condition, name)
        if condition is not None:
            conditions &= (df_files[name] == condition)

    df_select = df_files.loc[conditions, :].sort_values(by='id_file')
    df_select = df_select.drop_duplicates('id_file')

    images, metadata = grab_all_subtypes(df_select, exclude_one_slice)

    return (images, metadata, df_files)


def read_single_file(fp, images=[], metadata=[], exclude_one_slice=True):
    images, metadata = \
            read_cizlif_files_get_data(Path(fp),
                                       images=images,
                                       metadata=metadata,
                                       file_info=None,
                                       # {'id_file': id},
                                       exclude_one_slice=exclude_one_slice)
    return (images, metadata)


# %%
#

if __name__ == '__main__':
    data_path = Path('..', 'data')
    data_np_path = Path('.', 'data_numpy')

    comp = 'all'
    micro = 'Airyscan'
    cell = 'ES'
    cell_type = 'WT'

    # images, metadata, df_files = \
    #     read_files_pipeline(data_path,
    #                         microscope=micro,
    #                         cell=cell,
    #                         cell_type=cell_type,
    #                         comparison=comp)

    # files = list(data_path.rglob('*.czi'))
    # files.extend(list(data_path.rglob('*.lif')))

    # # exclude those starting with .
    # files = [file for file in files if not file.name.startswith('.')]
    # files = [file for file in files if not file.name.startswith('_')]

    # df_files = from_files_to_df(files)
