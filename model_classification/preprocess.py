from typing import Dict, Optional, Literal
import numpy as np
import scipy.signal
from scipy.signal import butter, iirnotch, filtfilt
import scipy.stats
import sklearn.decomposition

def preprocess_data(
    data: np.ndarray, 
    notch: bool = False, 
    highpass: bool = False, 
    bandpass: bool = False, 
    artifact_removal: bool = False, 
    normalize: bool = False
) -> np.ndarray:
    """
    Preprocesses input data by applying selected filters and transformations.

    Code modified from the Gumpy library as extracted from Roots et al. (2020).
    Source: https://github.com/rootskar/EEGMotorImagery 

    Args:
        data (ndarray): Input data to preprocess.
        notch_filter (bool): Apply notch filter if True.
        highpass_filter (bool): Apply highpass filter if True.
        bandpass_filter (bool): Apply bandpass filter if True.
        artifact_removal (bool): Remove artifacts if True.
        normalize (bool): Normalize data if True.

    Returns:
        ndarray: Preprocessed data after applying selected filters and transformations.
    """
    if notch:
        data = notch_filter(data)
    if highpass:
        data = highpass_filter(data)
    if bandpass:
        data = bandpass_filter(data)
    if normalize:
        data = normalize_data(data, 'min_max')
    if artifact_removal:
        data = remove_artifacts(data)

    return data

def notch_filter(
    data: np.ndarray, 
    axis: int = 0
) -> np.ndarray:
    """
    Apply a notch filter to data to suppress frequencies around an alternating current frequency.

    Args:
        data (ndarray): Input data to filter.
        axis (int, optional): Axis along which to filter the data. Default is 0.

    Returns:
        ndarray: Filtered data after applying the notch filter.
    """
    ac_freq = 60
    fs = 160  

    nyq = 0.5 * fs
    w0 = ac_freq / nyq
    Q = 30 
    b, a = scipy.signal.iirnotch(w0, Q)

    return scipy.signal.filtfilt(b, a, data, axis)

def highpass_filter(
    data: np.ndarray, 
    axis: int = 0
) -> np.ndarray:
    """
    Apply a highpass filter to data to attenuate frequencies below a specified cutoff frequency.

    Args:
        data (ndarray): Input data to filter.
        axis (int, optional): Axis along which to filter the data. Default is 0.

    Returns:
        ndarray: Filtered data after applying the highpass filter.
    """
    cutoff = 0.5  
    order = 4
    fs = 160  

    nyq = 0.5 * fs
    high = cutoff / nyq
    b, a = scipy.signal.butter(order, high, btype='highpass')

    return scipy.signal.filtfilt(b, a, data, axis)

def bandpass_filter(
    data: np.ndarray, 
    axis: int = 0
) -> np.ndarray:
    """
    Apply a bandpass filter to data to pass frequencies within a specified range.

    Args:
        data (ndarray): Input data to filter.
        axis (int, optional): Axis along which to filter the data. Default is 0.

    Returns:
        ndarray: Filtered data after applying the bandpass filter.
    """
    lowcut = 2  
    highcut = 60 
    order = 5
    fs = 160 

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='bandpass')

    return scipy.signal.filtfilt(b, a, data, axis)
    
def _norm_min_max(data):
    return (data - np.min(data))/(np.max(data)-np.min(data))

def _norm_mean_std(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev

def normalize_data(
    data: np.ndarray, 
    normalization_type: Literal['min_max', 'mean_std']
) -> np.ndarray:
    """
    Normalize data based on the specified normalization type.

    Args:
        data (ndarray): Input data to normalize.
        normalization_type (str): Type of normalization. Supported types are 'min_max' and 'mean_std'.

    Returns:
        ndarray: Normalized data with the same shape as input data.
    
    Raises:
        Exception: If an unsupported normalization type is provided.
    """
    norm_fns = {
        'min_max': _norm_min_max,
        'mean_std': _norm_mean_std
    }
    if normalization_type not in norm_fns:
        raise Exception(f"Normalization type '{normalization_type}' is not supported.")

    return norm_fns[normalization_type](data)

def remove_artifacts(
    data: np.ndarray, 
    n_components: Optional[int] = None, 
    check_result: bool = True
) -> np.ndarray:
    """
    Remove artifacts from data using Independent Component Analysis (ICA).

    Args:
        data (ndarray): Input data from which artifacts are to be removed.

    Returns:
        ndarray: Cleaned data after removing artifacts.
    """ 
    n_components = None  
    check_result = True 

    ica = sklearn.decomposition.FastICA(n_components)
    S_reconst = ica.fit_transform(data)
    A_mixing = ica.mixing_

    if check_result:
        assert np.allclose(data, np.dot(S_reconst, A_mixing.T) + ica.mean_)

    data = np.squeeze(S_reconst @ A_mixing)

    return data