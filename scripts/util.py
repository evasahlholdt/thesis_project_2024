import numpy as np
#import pickle
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

def combine_vectors(
    feature_vector: np.ndarray, attribute_vector: np.ndarray
    ) -> np.ndarray:
    ''' 
    Combine feature and attribute vectors. 

    Args: 
        - feature_vector: np.ndarray with features. 
        - attribute_vector: np.ndarray with attributes. 

    Returns: 
        - combined_data: np.ndarray with attributes and features. 
    '''

    run_length = 656 # config.max_sequence_len 
    
    reshaped_features = feature_vector.reshape(feature_vector.shape[0] * run_length, feature_vector.shape[2])  
    repeated_attributes = np.repeat(attribute_vector, run_length, axis=0)

    stacked_data = np.hstack((reshaped_features, repeated_attributes))

    combined_data = stacked_data.reshape(feature_vector.shape[0], run_length, stacked_data.shape[1])

    return combined_data

def define_attributes(
    real_attributes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    ''' 
    Extract unique variables present in attributes. 

    Args: 
        - real_attributes: np.ndarray with attributes in real data (str)

    Returns: 
        - subjects: np.ndarray with unique subjects. 
        - annotations: np.ndarray with unique annotations. 
    '''

    subjects = np.unique(real_attributes[:, -1])
    annotations = np.unique(real_attributes[:, -2])

    return subjects, annotations


def create_sample_subset(
    data: np.ndarray, subject: Optional[str] = None, annotation: Optional[str] = None
    ) -> np.ndarray:
    ''' 
    Extract samples for specific attributes.

    Args: 
        - data: np.ndarray containing features and attributes. 
        - subject: Optional (str). A specific subject from subjects attribute. 
        - annotation: Optional (str). A specific annotation from annotations attribute. 

    Returns: 
        - sample_subset: np.ndarray of samples corresponding to attribute set,
        only features (as floats).  
    '''

    if subject is not None and annotation is not None:
        mask = (data[:, 0, -1] == subject) & (data[:, 0, -2] == annotation)
    elif subject is not None:
        mask = (data[:, 0, -1] == subject)
    elif annotation is not None:
        mask = (data[:, 0, -2] == annotation)
    else:
        mask = np.ones(data.shape[0], dtype=bool)  

    sample_subset = data[mask]
    sample_subset = sample_subset[:, :, :-2]

    sample_subset = np.array(sample_subset, dtype=np.float64)

    return sample_subset

def save_results(
    results: Dict[str, float], 
    filename: str
    ) -> None:
    '''
    Save results. 

    Args: 
        results: dict. Dictionary containing results from optimization.
        dataset_name: str. Name for the model.
    '''
    os.makedirs('../results_optimization', exist_ok=True)

    filepath = os.path.join('../results_optimization', f"{file_name}.json")

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def load_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return json.load(file)
    return {}
