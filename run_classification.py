import numpy as np
import pickle
import os
from typing import Dict, Optional, Any
from tensorflow.keras import callbacks, backend as K
from tensorflow.keras.utils import to_categorical
from .model import ShallowConvNet
from .loader import load_data, load_eval_data

'''
Train and test ShallowConvNet. 

Be sure to select matching feature- and attribute vectors, as well as matching
training- and test data.
'''

def classify(
    dataset_name: str,
    train_feature_path: str, 
    train_attribute_path: str,
    test_feature_path: Optional[str] = None, 
    test_attribute_path: Optional[str] = None, 
    syn_feature_path: Optional[str] = None, 
    syn_attribute_path: Optional[str] = None,
    use_cpu: bool = True,
    preprocessing: bool = False,
    use_augmentation: bool = False,
    is_baseline: bool = False,
    augment_feature_path: Optional[str] = None,
    augment_attribute_path: Optional[str] = None,
    augment_percentage: Optional[int] = None
    ) -> None:

    ''' 
    Run classification. 

    Args: 
        train_feature_path: str. Path to train features.
        train_attribute_path: str. Path to train attributes. 
        test_feature_path: str. Path to test features.
        test_attribute_path: str. Path to test attributes.
        dataset_name: str. Name for dataset being tested.
        use_cpu: bool. True if CPU, False if GPU
        preprocessing: bool. Whether to apply preprocessing steps. 
        use_augmentation: bool. Whether to use data augmentation.
        is_baseline: bool. Whether data used for augmentation is baseline data.
        augment_feature_path: Optional[str]. Path to augmentation feature vector (.npy).
        augment_attribute_path: Optional[str]. Path to augmentation attribute vector (.npy).
        augment_percentage: Optional[int]. Percentage of data augmentation to use.
    ''' 

    if syn_feature_path and syn_attribute_path: 
        X_train, y_train, X_test, y_test = load_eval_data(
            train_feature_path,
            train_attribute_path,
            syn_feature_path,
            syn_attribute_path,
            preprocessing
        )
    else: 
        X_train, y_train = load_data(
            train_feature_path, 
            train_attribute_path,
            preprocessing, 
            use_augmentation, 
            is_baseline, 
            augment_feature_path, 
            augment_attribute_path, 
            augment_percentage)

        print(f"Shape of training features:", X_train.shape)
        print(f"Shape of training attributes:", y_train.shape)

        X_test, y_test = load_data(
            test_feature_path, 
            test_attribute_path, 
            preprocessing)

        print(f"Shape of testing features:", X_test.shape)
        print(f"Shape of testing attributes:", y_test.shape)

        print("Data loaded successfully.")
    
    if use_cpu:
        print("Using CPU")
        K.set_image_data_format('channels_last')
        sequence_length = X_train.shape[1]
        nr_channels = X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    else:
        print("Using GPU")
        K.set_image_data_format('channels_first')
        sequence_length = X_train.shape[2]
        nr_channels = X_train.shape[1]
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    
    nr_classes = 2
    y_train = to_categorical(y_train, nr_classes)
    y_test = to_categorical(y_test, nr_classes)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    model = ShallowConvNet(dataset_name, nr_channels, sequence_length, use_cpu = use_cpu)

    results = model.train_test(X_train, y_train, X_test, y_test)

    _save_results(results, dataset_name)

def _save_results(
    results: Dict[str, Any], 
    dataset_name: str
    ) -> None:
    '''
    Save results to folder classification_results.

    Args: 
        results: dict. Dictionary containing results.
        dataset_name: str. Name for the model.
    '''
    os.makedirs('../results/results_classification', exist_ok=True)

    filepath = os.path.join('../results/results_classification', f"results_{dataset_name}.pkl")

    with open(filepath, 'wb') as f: 
        pickle.dump(results, f)



