"""
Collection of function used throughout the scripts in this project.

They are split into: 
- Persistence:
    method for handling persistence diagrams and extracting vectorizations
    etc.
- Classification:
    method for classification and its associated data handling
-Clustering:
    methods used for the clustering pipeline
"""
# pyright: reportUnboundVariable=false

# %%
# import packages

import warnings
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as scistats

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from gudhi.representations import PersistenceImage, BettiCurve, Entropy

# type hints
from numpy.typing import ArrayLike
from typing import List, Callable
from typing import Union

import hashlib

warnings.filterwarnings("error")

# %%
# Persistence

def sanititze_persistencediagram(pers: ArrayLike,
        min_pers: float = 1e-15,
        set_infval: Union[None, float] = None) -> np.ndarray:
    """Sanitize the persistence diagram"""
    pers = np.array(pers)
    # handle the case where there is a pair with inf val
    if set_infval is None:
        pers = pers[pers[:,1] != np.inf, :]
    else:
        if isinstance(set_infval, (float, np.float32, np.float64)):
            pers[pers[:,1] == np.inf, 1] = set_infval
        else:
            raise ValueError('Invalid value for set_infval, needs to be None or a float')
    return pers

def sanitize_persistence(diags: List[np.ndarray],
        min_pers: float = 1e-15,
        set_infval: Union[None, float] = None) -> List[np.ndarray]:
    """Sanitize the persistence diagrams"""
    return [sanititze_persistencediagram(pers, min_pers, set_infval) for pers in diags]




def finite_max(arraymax: ArrayLike) -> np.ndarray:
    """Exclude infinite classes"""
    arraymax = arraymax[arraymax != np.inf]
    return arraymax


def tryskew(x: ArrayLike) -> np.ndarray:
    """ Try to compute skewness, if it does not work
    return np.nan
    """
    try:
        return scistats.skew(x)
    except RuntimeWarning:
        return np.nan


def trykurtosis(x: ArrayLike) -> np.ndarray:
    """ Try to compute kurtosis, if it does not work
    return np.nan
    """
    try:
        return scistats.kurtosis(x)
    except RuntimeWarning:
        return np.nan


def persistence_statistics(dgms: ArrayLike,
                           return_names: bool = False,
                           lifespan: bool = True):# -> Union(tuple(np.ndarray, List[str]), np.ndarray):
    """
    Returns a matrix with rows being the different persistence
    diagram statistics

    Parameters
    ----------
    dgms : _type_
        _description_
    return_names : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    statistics_names = ['min', 'max', 'range', 'mean', 'std', 'skew',
                        'kurtosis', 'mad',
                        '10quantile', '25quantile',
                        'median', '75quantile', '90quantile']
    statistics_funcs = [np.min,
                        np.max,
                        lambda x: np.max(x) - np.min(x),
                        np.mean,
                        np.std,
                        tryskew,
                        trykurtosis,
                        scistats.median_abs_deviation,
                        lambda x: np.quantile(x, q=0.1),
                        lambda x: np.quantile(x, q=0.25),
                        lambda x: np.quantile(x, q=0.5),
                        lambda x: np.quantile(x, q=0.75),
                        lambda x: np.quantile(x, q=0.9)
                        ]

    # if not(lifespan) and isinstance(dgms, list):
    #     dgms = np.array(dgms)
    stats = np.zeros([len(dgms), len(statistics_names)])
    for j, dgm in enumerate(dgms):
        if len(dgm) == 0:
            stats[j, :] = np.nan
        else:
            if lifespan:
                for i, sfunc in enumerate(statistics_funcs):
                    stats[j, i] = sfunc((dgm[:, 1] - dgm[:, 0]).ravel())
            else:
                for i, sfunc in enumerate(statistics_funcs):
                    stats[j, i] = sfunc(dgm.ravel())

    if return_names:
        return (stats, statistics_names)
    else:
        return stats


# we create a lambda function which removes the infinity bars from a barcode.
def replace_infinity(barcode: list[np.ndarray]) -> np.ndarray:
    """Replace the infinite bar with the one dying at the maximal value."""
    return (np.array([bars if bars[1] != np.inf
                     else [bars[0], np.max(barcode[barcode != np.inf])]
                     for bars in barcode]))


def remove_infinity(barcode: list[np.ndarray]) -> np.ndarray:
    """Remove the infinite bars completely."""
    return (np.array([bars for bars in barcode
                     if bars[1] != np.inf]))


def remove_low_persistence(dgms: list[np.ndarray], threshold: float=0) -> list[np.ndarray]:
    """Remove all persistence pairs whose persistence is below or equal threshold"""
    dgms = [np.asarray([bars for bars in dgm if bars[1] - bars[0] > threshold])
            for dgm in dgms]
    return (dgms)


def add_trivial_points(dgms: list[np.ndarray]) -> list[np.ndarray]:
    """Make sure that there is no empty persdistence diagram, other add trivial point."""
    num_nontrivial = np.sum([1 for x in dgms if x.size > 0])
    if num_nontrivial == 0:
        print('There is no non-empty persistence diagram!')
    minpers = np.min([np.min(x) for x in dgms if x.size > 0])
    return ([np.array([[minpers, minpers]]) if x.size == 0 else x
             for x in dgms])


# %%
# Classification


def compute_hash(array):
    # Ensure the array is in a consistent state
    array = np.ascontiguousarray(array)
    
    # Compute the hash
    hash_obj = hashlib.sha256(array)
    return hash_obj.hexdigest()

def same_size_training(labels: ArrayLike, size: int = 1,
                       train_size: float = 0.7,
                    #    data: Union(ArrayLike, None) = None,
                       data = None,
                       seed: int = 42, verbose: bool = False):
    # -> Union(tuple(ArrayLike, ArrayLike), List[ArrayLike],
    #          tuple(ArrayLike, ArrayLike, ArrayLike, ArrayLike)):
    """
    Generates a training-test split which has the same number of training
    samples from all the classes but the test set can contain different
    numbers of samples from each class.

    Parameters
    ----------
    labels : array_like of size (n,1)
        Labels upon which we base our splits.
    size : int, optional
        Number of splits to generate, by default 1
    data : list, optional
        data which we split along the same splits as the labesl,
        by default None
    train_size : float, optional
        Ratio of training data of the total number. , by default 0.7
    seed : int, optional
        Seed to be used in the split by numpy, by default 42
    verbose : bool, optional
        Print how many test samples are left after each split, by default False

    Returns
    -------
    _type_
        Outputs the either a tuple of train and test indices,
        or the list of such tuples if size > 1,
        or if data is provided, the train, test indices as well as train data and test data.
    """

    np.random.seed(seed)

    assert 0 < train_size and train_size < 1
    assert (data is None and size >= 1)

    if data is not None:
        size = 1

    traintest_list = []
    num_train_samples = None
    train_idx = None
    test_idx = None

    for _ in range(size):
        idx_classes = []
        for lbl in np.unique(labels):
            idx_class = np.where(labels == lbl)[0]
            np.random.shuffle(idx_class)

            idx_classes.append(np.copy(idx_class))

        num_train_samples = int(
            np.ceil(min([len(x) for x in idx_classes]) * train_size))

        train_idx = np.sort(
            np.hstack([x[:num_train_samples] for x in idx_classes]))
        test_idx = np.sort(
            np.hstack([x[num_train_samples:] for x in idx_classes]))

        traintest_list.append([train_idx.copy(), test_idx.copy()])

        if verbose:
            print('Test samples left: ',
                  [np.count_nonzero(labels == lbl) - num_train_samples
                   for lbl in np.unique(labels)])
    
    assert train_idx is not None

    if size > 1:
        hashes = np.array([[compute_hash(tt[0]), compute_hash(tt[1])] for tt in traintest_list])
        # now 
        delete_idx = []
        for i in range(len(hashes)):
            if hashes[i] in hashes[:i]:
                delete_idx.append(i)
        traintest_list = [traintest_list[i] for i in range(len(traintest_list)) if i not in delete_idx]
        del hashes

        if len(traintest_list) < size:
            print('There are less train-test splits than requested!')
    
    if data is None:
        if size == 1:
            return (train_idx, test_idx)
        else:
            return (traintest_list)
    else:
        return (train_idx, test_idx,
                [data[i] for i in train_idx],
                [data[i] for i in test_idx])


def drop_all_same(arr: ArrayLike, verbatim:bool = False):# -> tuple(np.ndarray, np.ndarray):
    """Drop all columns in arr which are the same for all rows.

    Parameters
    ----------
    arr : array_like
        Array of shape (n,m)
    verbatim : int, optional
        Print the number of clumns dropped, by default 0

    Returns
    -------
    tuple(array, array)
        Return the indices which are kept as well as the array
        where the columns which were the same are dropped
    """
    assert arr.ndim == 2
    idx_keep = np.where(~np.all(arr == arr[0, :], axis=0))[0]
    if verbatim:
        print(f'Dropped {np.shape(arr)[1] - len(idx_keep)}')
    return (idx_keep, arr[:, idx_keep])


def combine_data(key, all_data):
    """Combine different dimensions in the dictionary all_data
    for the same base keys.

    Parameters
    ----------
    key : list with 2 entries
        Gives [perstype, invariant] for which we 
        aggregate over the dimensions to get
        training and test data.

        if the second entry is itself a list, then
        do the above for each of these invariants in the list.
    all_data : dict
        Dictionary with keys being
        ('perstype', 'dimension', 'invariant', 'train' / 'test')

    Returns
    -------
    tuple of ndarrays
        Training data and test data
    """
    if not isinstance(key[1], list):
        X_train = np.hstack([all_data[(key[0], dim, key[1], 'train')]
                             for dim in range(3) if (key[0], dim, key[1], 'train') in all_data.keys()])
        X_test = np.hstack([all_data[(key[0], dim, key[1], 'test')]
                            for dim in range(3) if (key[0], dim, key[1], 'test') in all_data.keys()])

    else:
        X_train = np.hstack([all_data[(key[0], dim, keyi, 'train')]
                             for keyi in key[1] for dim in range(3) if (key[0], dim, keyi, 'train') in all_data.keys()])
        X_test = np.hstack([all_data[(key[0], dim, keyi, 'test')]
                            for keyi in key[1] for dim in range(3) if (key[0], dim, keyi, 'train') in all_data.keys()])

    return (X_train, X_test)


def grid_search_noscaling(X: np.ndarray,
                          y: np.ndarray,
                        #   n_components = None,
                        #   n_components: Union(None, int) = None,
                          verbose: int = 0,
                          disable_tqdm: bool = True,
                          n_jobs=-1) -> pd.DataFrame:
    """
    Perform classifciation for several different classifiers with
    possibly PCA dimensionality reduction using CrossValidation.
    No Scaling is done in the Pipeline! Pick the best one.

    Parameters
    ----------
    X: array of size (num_samples, num_features)
        Array giving the samples to perfom the classsifications on
    y: array of size (num_samples, 1)
        labels for doing the classification on
    n_components: None or int
        Number of components to keep. If n_components is None all components are kept:
        n_components == min(num_samples, num_features),
        by default None
    verbose: int
        Verbose is passed to GridSearchCV. Determines how much is
        printing from the intermediate steps.
        by default 0
    disable_tqdm: bool
        Detemines if to show the tqdm in the intermediate steps or not.
        by default True

    Returns
    -------
    GridSearchCV class
        _description_
    """
    min_features_samples = np.min(np.shape(X))

    # Determine the different steps in the pipeline.
    pipeline1 = Pipeline((
        ('clf', RandomForestClassifier()),
    ))

    pipeline1red = Pipeline((
        ('reduction', PCA()),
        ('clf', RandomForestClassifier()),
    ))

    pipeline2red = Pipeline((
        ('reduction', PCA()),
        ('clf', KNeighborsClassifier()),
    ))

    pipeline3 = Pipeline((
        ('clf', KNeighborsClassifier()),
    ))

    pipeline4 = Pipeline((
        ('clf', SVC()),
    ))

    pipeline4red = Pipeline((
        ('reduction', PCA()),
        ('clf', SVC()),
    ))

    parameters1 = {
        'clf__n_estimators': [10, 20, 30],
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [15],
        'clf__max_features': ['log2', 'sqrt', None]
    }

    parameters1red = {
        'clf__n_estimators': [10, 20, 30],
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [15],
        'clf__max_features': ['log2', 'sqrt', None]
    }

    parameters2red = {
        'clf__n_neighbors': [3, 7, 10],
        'clf__weights': ['uniform', 'distance']
    }

    parameters3 = {
        'clf__n_neighbors': [3, 7, 10],
        'clf__weights': ['uniform', 'distance']
    }

    parameters4 = {
        'clf__C': [0.01, 0.1, 1.0, 10],
        'clf__kernel': ['rbf', 'poly'],
        # 'clf__gamma': [0.01, 0.1, 1.0],
    }

    parameters4red = {
        'clf__C': [0.1, 1.0, 2],
        'clf__kernel': ['rbf', 'poly'],
        # 'clf__gamma': [0.01, 0.1, 1.0],
    }

    if min_features_samples > 2:
        parameters1red['reduction__n_components'] = [None, 2]
        parameters2red['reduction__n_components'] = [None, 2]
        parameters4red['reduction__n_components'] = [None, 2]

        if min_features_samples > 10:
            parameters1red['reduction__n_components'] = [None, 2, 10]
            parameters2red['reduction__n_components'] = [None, 2, 10]
            parameters4red['reduction__n_components'] = [None, 2, 10]
    
    pars = [parameters1,
            parameters1red,
            parameters2red,
            parameters3,
            parameters4,
            parameters4red
            ]
    pips = [pipeline1,
            pipeline1red,
            pipeline2red,
            pipeline3,
            pipeline4,
            pipeline4red
            ]

    if verbose > 0:
        print("starting Gridsearch")

    # TODO: make it a "proper" sklearn pipeline
    # this is a bit manual in the sense, that we could
    # also define a proper Pipeline directly in sklearn
    best_scores = []
    for i in tqdm(range(len(pars)), disable=disable_tqdm):
        gs = GridSearchCV(pips[i], pars[i],
                          scoring='accuracy', verbose=verbose, n_jobs=n_jobs)
        gs = gs.fit(X, y)
        if verbose > 1:
            print('Best score on Split:', gs.best_score_,
                  'Score on all:', gs.score(X, y))
        best_scores.append(gs.best_score_)

    besti = np.argmax(best_scores)
    gs = GridSearchCV(pips[besti], pars[besti],
                      scoring='accuracy', verbose=verbose, n_jobs=n_jobs)
    gs = gs.fit(X, y)

    if verbose > 0:
        print('Score:', gs.score(X, y))

    return gs

# %%
# Clustering methods

def majority_labels(labels_true: ArrayLike, labels_pred:ArrayLike,
                    function: Callable = metrics.accuracy_score,
                    function_b: Callable = metrics.normalized_mutual_info_score) -> ArrayLike:
    """Change labels from predicited labels to true labels such that
    we remove unnecessary labels. Maximizes the score in function.

    Parameters
    ----------
    labels_true : array-like vector
        True labels
    labels_pred : array-like vector
        Cluster labels, which we want to transform to adhere
        closer to the true labels.
    function : callable returing float, optional
        scoring method to be maximized, where higher score is better,
        by default metrics.accuracy_score
    function_b : callable returing float, optional
        scoring method to be maximized if function are the same, where higher score is better,
        by default metrics.accuracy_score

    Returns
    -------
    array_like (same size as labels_true)
        The new labels maximizing the function

    Yields
    ------
    _type_
        _description_

    Raises
    ------
    ValueError
        If we can't find a reasonable assignment.
    """

    assert np.all(np.shape(labels_true) == np.shape(labels_pred))

    num_clusters_pred = len(np.unique(labels_pred))
    clusters_true = np.unique(labels_true)

    # now go through all possible assignments to pick the
    # best one
    max_function = 0
    max_function_b = 0

    max_labels = None
    max_assignment = None
    for assignment in itertools.product(clusters_true,
                                        repeat=num_clusters_pred):
        # exclude the assignment if not all true clusters are represented
        if len(np.unique(assignment)) != len(clusters_true):
            continue

        # now do the assignment for the values in the array
        tmp_labels = np.zeros_like(labels_true)
        for i, lbl_pred in enumerate(np.unique(labels_pred)):
            tmp_labels[labels_pred == lbl_pred] = assignment[i]

        tmp_function = function(labels_true, tmp_labels)
        tmp_function_b = function_b(labels_true, tmp_labels)
        # now check if its maximal
        if max_function < tmp_function:
            max_function = tmp_function
            max_labels = tmp_labels.copy()
            max_assignment = assignment
            max_function_b = tmp_function_b

        elif max_function == tmp_function:
            if max_function_b < tmp_function_b:
                max_function = tmp_function
                max_labels = tmp_labels.copy()
                max_assignment = assignment
                max_function_b = tmp_function_b

    if ((len(np.unique(max_assignment)) != len(clusters_true))
            or (max_function == 0)):
        raise ValueError('The assignment did not yield an/a good assignment!')

    return max_labels

# %%
# Classification
def compute_vectorizations_all(labels, pers_all, resolution_pi=20, bandwidth=4, resolution_bc=250):
    assert 2 in pers_all.keys()
    assert 3 in pers_all.keys()

    data_df = [labels.reshape(-1,1)]
    data_df_cols = ['labels']

    # num_samples = len(lazbels)
    all_idx = np.arange(len(labels))

    for prefix in tqdm([2, 3]):
        ecc_train = []
        ecc_lens_train = []
        for dim in range(len(pers_all[prefix])):
            data = pers_all[prefix][dim]
            # data = [x[x[:, 1] != np.inf, :] for x in data]
            data = sanitize_persistence(data)

            ecc_train.extend([data[i] for i in all_idx])
            ecc_lens_train.append(len([data[i] for i in all_idx]))

            # for the persistence statistics we also include having the values encountered
            # in the persistence diagrams to be included as individual points
            persstats, colnames = persistence_statistics(data, return_names=True, lifespan=False)
            # persstats_ls, colnames_ls = persistence_statistics(data, return_names=True)
            pers_st_train = persstats[all_idx, :]

            # for entropy we mix the scalar version with its vector version
            # once for the unnormalized
            ent = Entropy(mode='vector', normalized=False)
            pers_ent_train = ent.fit_transform([data[i] for i in all_idx])
            ent = Entropy(mode='scalar', normalized=False)
            pers_ent_train = np.hstack([ent.fit_transform([data[i] for i in all_idx]),
                                    pers_ent_train])

            inv_bc = BettiCurve(resolution=resolution_bc)
            pers_bc_train = inv_bc.fit_transform([data[i] for i in all_idx])

            inv_pi = PersistenceImage(bandwidth=bandwidth,
                    resolution=[resolution_pi, resolution_pi])
            pers_pi_train = inv_pi.fit_transform([data[i] for i in all_idx])

            # exclude all which are the same for each sample in the training set
            idx_keep, pers_st_train = drop_all_same(pers_st_train)
            colnames = [colnames[i] for i in idx_keep]
            idx_keep, pers_bc_train = drop_all_same(pers_bc_train)
            idx_keep, pers_pi_train = drop_all_same(pers_pi_train)
            idx_keep, pers_ent_train = drop_all_same(pers_ent_train)

            for name_i, arr_i in zip(['stat', 'ent', 'bc', 'pi'],
                                    [pers_st_train, pers_ent_train,
                                    pers_bc_train, pers_pi_train]):
                if 'stat' in name_i:
                    columnnames = [f'{name_i}_{x}_{prefix}_{dim}' for x in colnames]
                else:
                    columnnames = [f'{name_i}_{i:04d}_{prefix}_{dim}' for i in range(np.shape(arr_i)[1])]
                
                assert len(columnnames) == np.shape(arr_i)[1]
                data_df_cols.extend(columnnames)
                data_df.append(arr_i)

        inv_ecc = BettiCurve(resolution=resolution_bc)
        ecc_train = inv_ecc.fit_transform(ecc_train)
        ecc_lens_train = np.cumsum(ecc_lens_train)
        if len(ecc_lens_train) == 2:
                ecc_train = ecc_train[:ecc_lens_train[0], :] \
                        - ecc_train[ecc_lens_train[0]:ecc_lens_train[1], :]
        elif len(ecc_lens_train) == 3:
                ecc_train = ecc_train[:ecc_lens_train[0], :] \
                        - ecc_train[ecc_lens_train[0]:ecc_lens_train[1], :] \
                        + ecc_train[ecc_lens_train[1]:, :]
        else:
                raise ValueError('Invalid number of dimensions')
        idx_keep, ecc_train = drop_all_same(ecc_train)
        
        data_df_cols.extend([f'ecc_{i:04d}_{prefix}' for i in range(np.shape(ecc_train)[1])])
        data_df.append(ecc_train)
    return pd.DataFrame(data=np.hstack(data_df), columns=data_df_cols)



def compute_vectorizations_traintest(invariant, prefix_dim, labels, pers_all, train_idx, test_idx,
            resolution_pi=20, bandwidth=4, resolution_bc=250, return_grid=False):
    assert 2 in pers_all.keys()
    assert 3 in pers_all.keys()
    assert invariant in ['stat', 'ent', 'bc', 'pi', 'ecc', 'all']
    assert prefix_dim in [2, 3]

    if return_grid:
        assert invariant in ['bc', 'ecc']

    X_train = []
    X_test = []
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    if invariant == 'all':
        X_train = []
        X_test = []

        for inv in ['stat', 'ent', 'bc', 'pi', 'ecc']:
            Xt, Xt2, _, _ = compute_vectorizations_traintest(inv, prefix_dim, labels,
                                    pers_all, train_idx, test_idx,
                                    resolution_pi, bandwidth, resolution_bc)
            X_train.append(Xt.copy())
            if len(test_idx) > 0:
                X_test.append(Xt2.copy())
        X_train = np.hstack(X_train)
        if len(test_idx) > 0:
            X_test = np.hstack(X_test)
        else:
            X_test = np.array([])

    elif invariant != 'ecc':
        for dim in range(len(pers_all[prefix_dim])):
            data = pers_all[prefix_dim][dim]
            # data = [x[x[:, 1] != np.inf, :] for x in data]
            data = sanitize_persistence(data)

            if invariant == 'stat': 
                persstats, _ = persistence_statistics(data, return_names=True, lifespan=False)
                
                pers_st_train = persstats[train_idx, :]
                if len(test_idx) > 0:
                    pers_st_test = persstats[test_idx, :]

                idx_keep, pers_st_train = drop_all_same(pers_st_train)
                if len(test_idx) > 0:
                    pers_st_test = pers_st_test[:, idx_keep]

                X_train.append(pers_st_train)
                if len(test_idx) > 0:
                    X_test.append(pers_st_test)

            elif invariant == 'ent':
                ent = Entropy(mode='vector', normalized=False)
                pers_ent_train = ent.fit_transform([data[i] for i in train_idx])
                if len(test_idx) > 0:
                    pers_ent_test = ent.transform([data[i] for i in test_idx])

                ent = Entropy(mode='scalar', normalized=False)
                pers_ent_train = np.hstack([ent.fit_transform([data[i] for i in train_idx]),
                                        pers_ent_train])
                if len(test_idx) > 0:
                    pers_ent_test = np.hstack([ent.transform([data[i] for i in test_idx]),
                                            pers_ent_test])

                idx_keep, pers_ent_train = drop_all_same(pers_ent_train)
                if len(test_idx) > 0:
                    pers_ent_test = pers_ent_test[:, idx_keep]

                X_train.append(pers_ent_train)
                if len(test_idx) > 0:
                    X_test.append(pers_ent_test)
            
            elif invariant == 'bc':
                inv_bc = BettiCurve(resolution=resolution_bc)
                pers_bc_train = inv_bc.fit_transform([data[i] for i in train_idx])
                if len(test_idx) > 0:
                    pers_bc_test = inv_bc.transform([data[i] for i in test_idx])

                idx_keep, pers_bc_train = drop_all_same(pers_bc_train)
                if len(test_idx) > 0:
                    pers_bc_test = pers_bc_test[:, idx_keep]

                X_train.append(pers_bc_train)
                if len(test_idx) > 0:
                    X_test.append(pers_bc_test)

                if return_grid:
                    grid = inv_bc.grid_
            
            elif invariant == 'pi':
                inv_pi = PersistenceImage(bandwidth=bandwidth,
                        resolution=[resolution_pi, resolution_pi])
                pers_pi_train = inv_pi.fit_transform([data[i] for i in train_idx])
                if len(test_idx) > 0:
                    pers_pi_test = inv_pi.transform([data[i] for i in test_idx])

                idx_keep, pers_pi_train = drop_all_same(pers_pi_train)
                if len(test_idx) > 0:
                    pers_pi_test = pers_pi_test[:, idx_keep]

                X_train.append(pers_pi_train)
                if len(test_idx) > 0:
                    X_test.append(pers_pi_test)
        
        X_train = np.hstack(X_train)
        if len(test_idx) > 0:
            X_test = np.hstack(X_test)
        else:
            X_test = np.array([])

    elif invariant == 'ecc':
        ecc_train = []
        ecc_lens_train = []
        ecc_test = []
        ecc_lens_test = []

        for dim in range(len(pers_all[prefix_dim])):
            data = pers_all[prefix_dim][dim]
            # data = [x[x[:, 1] != np.inf, :] for x in data]
            data = sanitize_persistence(data)

            ecc_train.extend([data[i] for i in train_idx])
            ecc_lens_train.append(len([data[i] for i in train_idx]))

            if len(test_idx) > 0:
                ecc_test.extend([data[i] for i in test_idx])
                ecc_lens_test.append(len([data[i] for i in test_idx]))

        inv_ecc = BettiCurve(resolution=resolution_bc)
        ecc_train = inv_ecc.fit_transform(ecc_train)

        if return_grid:
            grid = inv_ecc.grid_

        ecc_lens_train = np.cumsum(ecc_lens_train)
        if len(ecc_lens_train) == 2:
                ecc_train = ecc_train[:ecc_lens_train[0], :] \
                        - ecc_train[ecc_lens_train[0]:ecc_lens_train[1], :]
        elif len(ecc_lens_train) == 3:
                ecc_train = ecc_train[:ecc_lens_train[0], :] \
                        - ecc_train[ecc_lens_train[0]:ecc_lens_train[1], :] \
                        + ecc_train[ecc_lens_train[1]:, :]
        else:
                raise ValueError('Invalid number of dimensions')
        idx_keep, X_train = drop_all_same(ecc_train)

        if len(test_idx) > 0:
            ecc_test = inv_ecc.transform(ecc_test)
            
            ecc_lens_test = np.cumsum(ecc_lens_test)
            if len(ecc_lens_test) == 2:
                    ecc_test = ecc_test[:ecc_lens_test[0], :] \
                            - ecc_test[ecc_lens_test[0]:, :]
            elif len(ecc_lens_test) == 3:
                    ecc_test = ecc_test[:ecc_lens_test[0], :] \
                            - ecc_test[ecc_lens_test[0]:ecc_lens_test[1], :] \
                            + ecc_test[ecc_lens_test[1]:, :]
            else:
                    raise ValueError('Invalid number of dimensions')
            X_test = ecc_test[:, idx_keep]
        else:
            X_test = np.array([])
    else:
        raise ValueError('Invalid invariant')
    
    if return_grid and invariant in ['bc', 'ecc']:
        return X_train, X_test, y_train, y_test, grid
    else:
        return X_train, X_test, y_train, y_test
