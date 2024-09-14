import numpy as np

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

    run_length = 656
    
    reshaped_features = feature_vector.reshape(feature_vector.shape[0] * run_length, feature_vector.shape[2])  
    repeated_attributes = np.repeat(attribute_vector, run_length, axis=0)

    stacked_data = np.hstack((reshaped_features, repeated_attributes))

    combined_data = stacked_data.reshape(feature_vector.shape[0], run_length, stacked_data.shape[1])

    return combined_data
