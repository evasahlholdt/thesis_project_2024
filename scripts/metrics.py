import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
from dtaidistance.dtw_ndim import distance as multi_dtw_distance
from statsmodels.tsa import stattools
from collections import Counter, defaultdict
from scripts.util import define_attributes

def mean_std(
    real_features_path: str, 
    syn_features_path: str, 
    dataset_name: str):
    '''
    Compute mean and standard deviation for real and synthetic features.
    
    Args: 
        real_features_path: Path to .npy file with real features.
        syn_features_path: Path to .npy file with synthetic features. 
        dataset_name: Name of dataset version being used. 
    '''
    real_features = np.load(real_features_path, allow_pickle=True)
    synthetic_features = np.load(syn_features_path, allow_pickle=True)
    
    number_of_samples, seq_len, dim = real_features.shape  

    for i in range(number_of_samples):
        if i == 0:
            prep_data = np.reshape(np.mean(real_features[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(synthetic_features[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data, 
                                        np.reshape(np.mean(real_features[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat, 
                                            np.reshape(np.mean(synthetic_features[i, :, :], 1), [1, seq_len])))
    
    prep_data = prep_data.flatten()
    prep_data_hat = prep_data_hat.flatten()

    real_mean = np.mean(prep_data)
    real_std = np.std(prep_data)
    syn_mean = np.mean(prep_data_hat)
    syn_std = np.std(prep_data_hat)

    print(f"Mean and standard deviation for dataset {dataset_name}:")
    print(f"Real features, mean:", real_mean)
    print(f"Real features, std:", real_std)
    print(f"Synthetic features, mean:", syn_mean)
    print(f"Synthetic features, std:", syn_std)


def euclidean_distance(
    real_sample: np.ndarray, 
    syn_sample: np.ndarray
) -> float:
    ''' 
    Calculate the Euclidean Distance (ED) between multivariate time series. 

    Args: 
        - real_sample: np.ndarray with a real sample (features only)
        - syn_sample: np.ndarray with a synthetic sample (features only)
    
    Returns: 
        - ed: ED across dimensions.
    '''

    # assert real_sample.shape == syn_sample.shape, "Input arrays must have the same shape"

    # ed = np.linalg.norm(real_sample.flatten() - syn_sample.flatten())
    
    # return ed.item()
    
    assert real_sample.shape == syn_sample.shape, "Input arrays must have the same shape"

    n_features = real_sample.shape[-1]
    distances = []

    for i in range(n_features):
        ed = np.linalg.norm(real_sample[:, i] - syn_sample[:, i])
        distances.append(ed)

    ed_distance = np.mean(distances) / n_features
    
    return ed_distance.item() 

def feature_correlation(real_data: np.ndarray, syn_data: np.ndarray
) -> np.ndarray:
    '''
    Compute average correlation matrices (PCC) for real and synthetic data. 

    Create squared error matrix for each feature pair, mean differences,
     and mean squared error across features. 

    Args: 
        - real_data: np.ndarray with real data (features only, several samples)
        - syn_sample: np.ndarray with synthetic data (features only, several samples)
    
    Returns: 
        - feature_matrices: np.ndarray. Correlation matrices for real and synthetic 
        data and squared error matrix. 
        - mean_correlation_real: Mean PCC for baseline data. 
        - mean_correlation_syn: Mean PCC for synthetic data. 
        - mean_difference: Difference between mean_correlation_real and mean_correlation_syn.
        - mean_squared_error: Mean squared error across feature correlations.
    '''

    temp_matrices_real = np.zeros((real_data.shape[0], real_data.shape[2], real_data.shape[2]))
    temp_matrices_syn = np.zeros((syn_data.shape[0], syn_data.shape[2], syn_data.shape[2]))

    for i in range(real_data.shape[0]):
        temp_matrices_real[i] = np.corrcoef(real_data[i], rowvar=False)
        
    for j in range(syn_data.shape[0]):
        temp_matrices_syn[j] = np.corrcoef(syn_data[j], rowvar=False)

    correlation_matrix_real = np.mean(temp_matrices_real, axis=0)
    correlation_matrix_syn = np.mean(temp_matrices_syn, axis=0)

    mean_correlation_real = np.mean(correlation_matrix_real[~np.eye(correlation_matrix_real.shape[0], dtype=bool)])
    mean_correlation_syn = np.mean(correlation_matrix_syn[~np.eye(correlation_matrix_syn.shape[0], dtype=bool)])

    mean_difference = mean_correlation_real - mean_correlation_syn

    squared_error_matrix = np.square(correlation_matrix_real - correlation_matrix_syn)

    mean_squared_error = np.mean(squared_error_matrix[~np.eye(squared_error_matrix.shape[0], dtype=bool)])

    feature_matrices = {'correlation_matrix_real': correlation_matrix_real, 'correlation_matrix_syn': correlation_matrix_syn, 'squared_error_matrix': squared_error_matrix }

    return feature_matrices, mean_difference.item(), mean_squared_error.item()

def auto_correlation(sample: np.ndarray, k: Optional[int] = None
) -> np.ndarray:
    """
    Compute autocorrelation for a given sample. 

    Computes autocorrelation for each feature, then aggregates.

    Args: 
        - sample (np.ndarray): Real or synthetic sample shape [seq_len, features].
        - k (int): Maximum number of lags to compute ACF for. Default to sequence
        lenght - 1. 

    Returns: 
        - Mean ACF for sample, [k, features]
    """
    if k is None: 
        k = round(sample.shape[0]//10)
        #k = sample.shape[0]-1
    
    features = sample.shape[1]
    acf_features = []

    for i in range(features):
        sample_feature = sample[:,i]
        acf = stattools.acf(sample_feature, nlags=k, fft=True)
        acf_features.append(acf)

    mean_acf = np.mean(acf_features, axis=0)

    return mean_acf

def hellinger_distance(real_attribute_vector: np.ndarray, syn_attribute_vector: np.ndarray):
    ''' 
    Compute Hellinger distance to assess joint attribute distributions.
    '''
    def hellinger_distance(p, q):
        ''' 
        Utility function to calculate Hellinger distance.
        '''
        p = np.asarray(p)
        q = np.asarray(q)
        p = p / np.sum(p)
        q = q / np.sum(q)
        distance = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)
        return distance

    def normalized_value_counts(array, all_combinations):
        ''' 
        Utility function to normalize value counts for joint distributions.
        '''
        total_count = len(array)
        value_counts = Counter(array)
        normalized_counts = np.array([value_counts.get(comb, 0) / total_count for comb in all_combinations])
        return normalized_counts

    real_tuples = [tuple(map(str, row)) for row in real_attribute_vector]
    syn_tuples = [tuple(map(str, row)) for row in syn_attribute_vector]

    all_combinations = sorted(set(real_tuples).union(set(syn_tuples)))

    real_counts = normalized_value_counts(real_tuples, all_combinations)
    syn_counts = normalized_value_counts(syn_tuples, all_combinations)

    distance = hellinger_distance(real_counts, syn_counts)

    return distance

def attribute_class_error(real_attribute_vector: np.ndarray, syn_attribute_vector: np.ndarray):
    ''' 
    Compute attribute class error to assess attribute distribution.
    '''

    subjects, annotations = define_attributes(real_attribute_vector)

    attribute_error_dict = {}

    for subject in subjects:
            for annotation in annotations:

                real_subset = []
                real_subset.extend(real_attribute_vector[(real_attribute_vector[:, 1] == subject) & (real_attribute_vector[:, 0] == annotation)])
                real_subset = np.array(real_subset)
                syn_subset = []
                syn_subset.extend(syn_attribute_vector[(syn_attribute_vector[:, 1] == subject) & (syn_attribute_vector[:, 0] == annotation)])
                syn_subset = np.array(syn_subset)

                if syn_subset.shape[0] == 0:

                    attribute_error_dict[(subject, annotation)] = {
                        'attribute_class_error': "No synthetic data for this class",
                        'n_syn_samples': syn_subset.shape[0],
                        'n_real_samples': real_subset.shape[0]
                    }

                else:
                    
                    attribute_error_dict[(subject, annotation)] = {
                        'attribute_class_error': ((syn_subset.shape[0] - real_subset.shape[0]) / real_subset.shape[0]) * 100,
                        'n_syn_samples': syn_subset.shape[0],
                        'n_real_samples': real_subset.shape[0]
                    }
    
    return attribute_error_dict

def stats_attributes(
    attribute_path: str,
    dataset_name: str
):
    attributes = np.load(attribute_path, allow_pickle=True)

    subject_ids = attributes[:, 1]
    annotations = attributes[:, 0]

    counts_per_subject = defaultdict(lambda: defaultdict(int))

    for annotation, subject_id in zip(annotations, subject_ids):
        counts_per_subject[subject_id][annotation] += 1

    counts_list = []

    for subject_id, annotation_counts in counts_per_subject.items():
        for annotation, count in annotation_counts.items():
            counts_list.append((annotation, count))

    counts_df = pd.DataFrame(counts_list, columns=["annotation", "count"])

    mean_counts = counts_df["count"].mean()
    std_counts = counts_df["count"].std()

    stats_per_annotation = counts_df.groupby('annotation')['count'].agg(['mean', 'std'])

    print(f"Dataset: {dataset_name}")
    for annotation, row in stats_per_annotation.iterrows():
        print(f"Annotation: {annotation}, Mean: {row['mean']:.2f}, Std: {row['std']:.2f}")