import numpy as np
import pandas as pd
import pickle
from typing import Optional
import matplotlib.pyplot as plt
import random
from scipy.interpolate import CubicSpline
import os

def create_baseline(
    real_features: np.ndarray,
    feature_name: str, 
    version_name: str,
    noise_mean: Optional[float] = 0.0, 
    noise_std: Optional[float] = 5.0, 
    num_knots: Optional[int] = 32, 
    warp_std_dev: Optional[float] = 0.2
) -> np.ndarray:
    ''' 
    Apply transformations and save new vectors. 
    
    Args: 
        real_features: np.ndarray. Time series real_features of shape (n_samples, seq_len, num_features)
        feature_name: str. Name of real_features vector
        version_name: str. Name of baseline version relating to the amount of noise
        noise_mean (Optional): float. Default = 0 
        noise_std (Optional): float. Default = 5 
        num_knots (Optional): int. Default = 32
        warp_std_dev (Optional): float. Default = 0.2

    Returns: 
        transformed_features: np.ndarray. Transformed features
    '''
    jittered_features = _jittering(real_features, noise_mean, noise_std)
    transformed_features = _magnitude_warping(jittered_features, num_knots, warp_std_dev)

    _save_data(transformed_features, feature_name, version_name)

    print(f"Applied transformations to {feature_name}")

def _save_data(
    transformed_features: np.ndarray, 
    feature_name: str, 
    version_name: str
) -> None: 
    ''' 
    Save baseline vector.
    '''

    os.makedirs('../data_synthetic/baseline', exist_ok=True)

    base_name = feature_name.split('train_features_')[-1]

    path = os.path.join('../data_synthetic/baseline', f"train_features_{base_name}_{version_name}.npy")

    np.save(path, transformed_features)

def _jittering(
    real_features: np.ndarray, 
    noise_mean: float, 
    noise_std: float
) -> np.ndarray:
    ''' 
    Add Gaussian noise to real features.

    Args: 
        real_features: np.ndarray. Time series real_features of shape (n_samples, seq_len, num_features)
        noise_mean: Mean for Gaussian distribution to sample noise from
        noise_std: Standard deviation of Gaussian distribution to sample noise from 

    Returns: 
        np.ndarray: transformed features
    '''
    
    noise = np.random.normal(loc=noise_mean, scale=noise_std, size=(real_features.shape[0], real_features.shape[1], real_features.shape[2]))
    noisy_features = real_features + noise
    return noisy_features

def _magnitude_warping(
    real_features: np.ndarray, 
    num_knots: int, 
    warp_std_dev: float
) -> np.ndarray:
    '''
    Applies magnitude warping to multiple real_features of a multi-dimensional time series using cubic splines.

    Args: 
        real_features: np.ndarray. Time series real_features of shape (n_samples, seq_len, num_features)
        num_knots: int. Number of control points for splines
        warp_std_dev: float. Standard deviation for distorting the values of control points

    Returns: 
        np.ndarray: transformed features
    '''
    n_samples, seq_len, num_features = real_features.shape
    warped_real_features = []

    for sample in real_features:
        warped_sample = sample.copy()

        for feature_idx in range(num_features):
            time_series = sample[:, feature_idx]
            
            knot_positions = np.linspace(0, len(time_series) - 1, num=num_knots)
            knot_values = 1 + np.random.normal(0, warp_std_dev, num_knots)

            spline = CubicSpline(knot_positions, knot_values)

            time_indexes = np.arange(len(time_series))

            warped_time_series = time_series * spline(time_indexes)

            warped_sample[:, feature_idx] = warped_time_series
        
        warped_real_features.append(warped_sample)

    return np.array(warped_real_features)