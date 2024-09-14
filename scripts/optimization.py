import argparse
import numpy as np
import pandas as pd
import pickle
import os
import json
from typing import Callable, Dict, List, Optional, Tuple, Any
import warnings
import torch
from sklearn.model_selection import ParameterGrid
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")

from model_doppelganger.dgan import DGAN
from model_doppelganger.config import DGANConfig, OutputType, Normalization

def run_optimization(train_feature_path: str, train_attribute_path: str, epochs: Optional[int] = 100
) -> None: 
    '''
    Run grid search for optimization. 

    Args: 
        train_feature_path: str. 
        train_attribute_path: str. 

    '''

    train_features, train_attributes = _load_data(train_feature_path, train_attribute_path)

    param_grid = {
        'sample_len': [4, 8, 16, 41], 
        'batch_size': [8, 16, 32], 
        'generator_learning_rate': [1e-3, 1e-4, 1e-5, 1e-6], 
        'discriminator_learning_rate': [1e-3, 1e-4, 1e-5], 
        }

    param_list = list(ParameterGrid(param_grid))
    print(f"Number of hyperparameter configurations to search:", len(param_list))

    optimization_results = _load_results('/work/GAN/results_optimization/optimization_results.json')

    counter = len(optimization_results)

    attribute_noise_vector = torch.tensor(np.random.randn(train_attributes.shape[0], 10), dtype=torch.float32)
    feature_noise_vector = torch.tensor(np.random.randn(train_features.shape[0], train_features.shape[1], 10), dtype=torch.float32)

    for params in param_list[counter:]:
        model = DGAN(DGANConfig(
            max_sequence_len=train_features.shape[1],
            apply_feature_scaling=True,
            apply_example_scaling=False,
            normalization=Normalization.MINUSONE_ONE,
            
            sample_len=params['sample_len'],
            batch_size=params['batch_size'],
            generator_learning_rate=params['generator_learning_rate'],
            discriminator_learning_rate=params['discriminator_learning_rate'],
            
            epochs=epochs,
        ),
            fixed_attribute_noise=attribute_noise_vector,
            fixed_feature_noise=feature_noise_vector,
            attributes=train_attributes,
            features=train_features
        )

        losses, metrics = model.train_numpy(optimization=True)

        optimization_results[counter] = {
            'params': params,
            'losses': losses,
            'metrics': metrics
        }

        _save_results('optimization_results', optimization_results) 

        counter += 1

        print(f"Finished training DG for parameter configuration {counter}")

    print("Finished grid search.")

def _load_data(
    feature_path: str, 
    attribute_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
    ''' 
    Load features and attributes for training.

    Args: 
        feature_path: str. Path to features file .npy
        attribute_path: str. Path to attributes file .npy

    Returns: 
        train_features: np.ndarray. Training features
        train_attributes: np.ndarray. Training attributes
    ''' 
    train_features = np.load(fr'{feature_path}', allow_pickle=True)
    train_attributes = np.load(fr'{attribute_path}', allow_pickle=True)

    print("Shape of feature vector:", train_features.shape)
    print("Shape of attribute vector:", train_attributes.shape)

    return train_features, train_attributes

def _load_results(file_path: str) -> Dict:
    ''' 
    Load results from optimization if existing; otherwise initialize 
    empty dictionary to store results.
    '''
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

def _save_results(
    results: Dict[str, float], 
    filename: str
    ) -> None:
    '''
    Save results. 

    Args: 
        results: dict. Dictionary containing results from optimization.
        dataset_name: str. Name for the model.
    '''
    os.makedirs('../results/results_optimization', exist_ok=True)

    filepath = os.path.join('../results/results_optimization', f"{file_name}.json")

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Run grid search for optimization.")
    parser.add_argument('--train_feature_path', type=str, required=True, help="Path to the training features file .npy")
    parser.add_argument('--train_attribute_path', type=str, required=True, help="Path to the training attributes file .npy")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    
    args = parser.parse_args()
    
    run_optimization(args.train_feature_path, args.train_attribute_path, args.epochs)

if __name__ == "__main__":
    main()
