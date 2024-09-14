import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from .preprocess import preprocess_data
from .utils import combine_vectors

def load_data(
    feature_path: str, 
    attribute_path: str, 
    preprocessing: bool = False,
    use_augmentation: bool = False,
    is_baseline: bool = False,
    augment_feature_path: Optional[str] = None,
    augment_attribute_path: Optional[str] = None,
    augment_percentage: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load feature and attribute vectors from path and transform to required input format for classification. 

    If preprocessing = True, preprocessing steps according to Roots et al. (2020) are performed, 
    see https://www.mdpi.com/2073-431X/9/3/72 

    Args: 
        features: str. Path to feature vector (.npy)
        attributes: str. Path to attribute vector (.npy)
        preprocessing: bool. Whether to apply preprocessing steps. 
        use_augmentation: bool. Whether to use data augmentation.
        is_baseline: bool. Whether data used for augmentation is baseline data.
        augment_feature_path: Optional[str]. Path to augmentation feature vector (.npy).
        augment_attribute_path: Optional[str]. Path to augmentation attribute vector (.npy).
        augment_percentage: Optional[int]. Percentage of data augmentation to use.

    Returns: 
        X: np.ndarray. Features (EEG signals) for classification. 
        y: np.ndarray. Left/right hand imagery labels for classification. 
    '''

    features = np.load(fr'{feature_path}', allow_pickle=True)
    attributes = np.load(fr'{attribute_path}', allow_pickle=True)

    data = combine_vectors(features, attributes)
    data = data[:, :, :-1] # Drop subject_ID
    label_mapping = {'left': 0, 'right': 1}
    data[:, :, -1] = np.vectorize(label_mapping.get)(data[:, :, -1])
    data = data[:, :640, :] # Truncate to 640 measurements according to Roots et al. (2020)
    
    if use_augmentation:
        assert augment_feature_path and augment_attribute_path and augment_percentage is not None, "Must provide augment_feature_path, augment_attribute_path and augment_percentage"
        augment_features = np.load(fr'{augment_feature_path}', allow_pickle=True)
        augment_attributes = np.load(fr'{augment_attribute_path}', allow_pickle=True)
        data_augment = combine_vectors(augment_features, augment_attributes)

        if is_baseline:
            np.random.shuffle(data_augment)

        if augment_percentage == 100:
            pass
        else:
            assert 0 <= augment_percentage <= 100, "augment_percentage must be integer between 0 and 100"

            num_samples = int(len(data_augment) * (augment_percentage / 100.0))

            data_augment = data_augment[np.random.choice(len(data_augment), num_samples, replace=False)]
        
        data_augment = data_augment[:, :, :-1] # Drop subject_ID
        label_mapping = {'left': 0, 'right': 1}
        data_augment[:, :, -1] = np.vectorize(label_mapping.get)(data_augment[:, :, -1])
        data_augment = data_augment[:, :640, :]

        data = np.concatenate((data, data_augment), axis=0)

    X = data[:, :, :-1]
    y = data[:, 0, -1:]

    if preprocessing: 
        for j in range(X.shape[0]):
            for i in range(X.shape[2]):
                channel_data = X[j, :, i]
                channel_data = preprocess_data(channel_data, 
                                                notch=True,
                                                highpass=False, 
                                                bandpass=True,
                                                artifact_removal=False, 
                                                normalize=False)
                X[j, :, i] = channel_data
    else: 
        pass

    return X, y

def load_eval_data(
    real_feature_path: str, 
    real_attribute_path: str,
    syn_feature_path: str,
    syn_attribute_path: str,
    preprocessing: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load feature and attribute vectors from path and transform to required input format for classification. 


    Args: 
        features: str. Path to feature vector (.npy)
        attributes: str. Path to attribute vector (.npy)

    Returns: 
        X: np.ndarray. Features (EEG signals) for classification. 
        y: np.ndarray. Left/right hand imagery labels for classification. 
    '''

    real_features = np.load(fr'{real_feature_path}', allow_pickle=True)
    real_attributes = np.load(fr'{real_attribute_path}', allow_pickle=True)
    syn_features = np.load(fr'{syn_feature_path}', allow_pickle=True)
    syn_attributes = np.load(fr'{syn_attribute_path}', allow_pickle=True)

    real_data = combine_vectors(real_features, real_attributes)
    syn_data = combine_vectors(syn_features, syn_attributes)

    # add labels
    real_array = np.full((real_data.shape[0], real_data.shape[1], 1), 0, dtype=int)  # 0 for real
    syn_array = np.full((syn_data.shape[0], syn_data.shape[1], 1), 1, dtype=int) # 1 for synthetic

    real_data = np.concatenate((real_data, real_array), axis=2)
    syn_data = np.concatenate((syn_data, syn_array), axis=2)

    data = np.concatenate((real_data, syn_data), axis=0)

    print(f"Real data shape:", real_data.shape)
    print(f"Syn data shape:", syn_data.shape)
    print(f"Combined shape:", data.shape)

    # Sample 80 % for training, 20 % for testing
    np.random.seed(42)

    indices = np.random.permutation(data.shape[0])

    train_data = data[indices[:(int(0.8 * data.shape[0]))]]
    test_data = data[indices[(int(0.8 * data.shape[0])):]]

    train_data = train_data[:, :640, :] 
    test_data = test_data[:, :640, :]

    X_train = train_data[:, :, :-3]
    y_train = train_data[:, 0, -1:] 
    X_test = test_data[:, :, :-3] 
    y_test = test_data[:, 0, -1:] 

    print(f"X_train shape: ", X_train.shape)
    print(f"y_train shape:", y_train.shape)
    print(f"X_test shape: ", X_test.shape)
    print(f"y_test shape:", y_test.shape)

    print("First example in X_train: ", X_train[0, :])
    print("First example in y_train:", y_train)
    print("First example in X_test: ", X_test[0, :])
    print("First example in y_test:", y_test)

    if preprocessing: 
        for j in range(X_train.shape[0]):
            for i in range(X_train.shape[2]):
                train_channel_data = X_train[j, :, i]
                train_channel_data = preprocess_data(train_channel_data, 
                                                notch=True,
                                                highpass=False, 
                                                bandpass=True,
                                                artifact_removal=False, 
                                                normalize=False)
                X_train[j, :, i] = train_channel_data
        for j in range(X_test.shape[0]):
            for i in range(X_test.shape[2]):
                test_channel_data = X_test[j, :, i]
                test_channel_data = preprocess_data(test_channel_data, 
                                                notch=True,
                                                highpass=False, 
                                                bandpass=True,
                                                artifact_removal=False, 
                                                normalize=False)
                X_test[j, :, i] = test_channel_data
    else: 
        pass

    return X_train, y_train, X_test, y_test