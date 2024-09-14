import argparse
import numpy as np
import json
import pickle
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch

## Append parent directory
sys.path.append(os.path.dirname(os.getcwd()))

from model_doppelganger.dgan import DGAN
from model_doppelganger.config import DGANConfig, OutputType, Normalization

# When training using CPU
import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")

'''
Script for training DG and generating data. 

To run this script, follow the steps (in terminal): 
-   cd path_to_project_folder
-   pip install -r requirements.txt
-   python train_generate.py --args (e.g. --epochs 1)

If you are running on a VM, you might need to first install pip: 
-   sudo apt install pip

If you want to check if GPUs are available, ensure that you have installed 
the requirements (specifically torch) and run: 
-   torch.cuda.is_available()

NB: 
DoppelGANger (DG) is a time-consuming model. Expect long training time. 
'''

def train_generate(
    dataset_name: str, 
    real_feature_path: str, 
    real_attribute_path: str, 
    epochs: int
    ) -> None:
    '''
    Train DG and generate data. 

    Args:
        dataset_name: str. Name for trained model
        real_feature_path: str. Path to real_feature_path.npy
        real_attribute_path: str. Path to real_attribute_path.npy
        epochs: int. Number of epochs for training.
    '''

    train_features, train_attributes = _load_data(real_feature_path, real_attribute_path)

    print("Data loaded successfully.")

    nr_samples = train_features.shape[0]

    attribute_noise_vector = torch.tensor(np.random.randn(train_attributes.shape[0], 10), dtype=torch.float32)
    feature_noise_vector = torch.tensor(np.random.randn(train_features.shape[0], train_features.shape[1], 10), dtype=torch.float32)

    model = DGAN(DGANConfig(
        max_sequence_len=656, 
        sample_len=4, 
        batch_size=8,
        apply_feature_scaling=True, 
        use_attribute_discriminator=False, 
        apply_example_scaling=False,
        normalization=Normalization.MINUSONE_ONE, 
        generator_learning_rate=1e-4,
        discriminator_learning_rate=1e-4,
        epochs=epochs,
    ),
        fixed_attribute_noise=attribute_noise_vector,
        fixed_feature_noise=feature_noise_vector,
        attributes=train_attributes,
        features=train_features
    )

    losses, metrics = model.train_numpy(optimization=False)

    print("Training completed")

    _save_info(losses, metrics, dataset_name)

    print(f"Saved training_info with name training_info_{dataset_name}")

    print("Generating data...")

    synthetic_attributes, synthetic_features = model.generate_numpy(nr_samples, attribute_noise_vector, feature_noise_vector)

    _save_data(synthetic_attributes, synthetic_features, dataset_name)

    print(f"Saved synthetic attributes to data_synthetic/DG_generated with name synthetic_attributes_{dataset_name}")
    print(f"Saved synthetic features to data_synthetic/DG_generated with name synthetic_features_{dataset_name}")


def _load_data(
    real_feature_path: str, 
    real_attribute_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
    ''' 
    Load real features and attributes for training.

    Args: 
        real_feature_path: str. Path to real_feature_path.npy
        real_attribute_path: str. Path to real_attribute_path.npy

    Returns: 
        train_features: np.ndarray. Training features
        train_attributes: np.ndarray. Training attributes
    ''' 
    train_features = np.load(fr'{real_feature_path}', allow_pickle=True)
    train_attributes = np.load(fr'{real_attribute_path}', allow_pickle=True)

    return train_features, train_attributes

def _save_info(
    losses: Dict[str, float], 
    metrics: Dict[str, float], 
    dataset_name: str
    ) -> None:
    '''
    Save loss, MV-DTW and Hellinger distances calculated during training. 

    Args: 
        losses: dict. Dictionary containing loss values.
        metrics: dict. Dictionary containing metrics values.
        dataset_name: str. Name for the model.
    '''
    os.makedirs('../results/results_training_info', exist_ok=True)

    training_info = {
        'losses': losses,
        'metrics': metrics
    }

    filepath = os.path.join('../results/results_training_info', f"training_info_{dataset_name}.json")

    with open(filepath, 'w') as f:
        json.dump(training_info, f, indent=4)

def _save_data(
    synthetic_attributes: np.ndarray, 
    synthetic_features: np.ndarray, 
    dataset_name: str
    ) -> None: 
    '''
    Save synthetic attributes and features to folder /data_synthetic/DG_generated.

    Args:
        synthetic_attributes (np.ndarray): Array containing synthetic attributes.
        synthetic_features (np.ndarray): Array containing synthetic features.
        dataset_name (str): Name of the dataset being saved.
    '''

    os.makedirs('../data_synthetic/DG_generated', exist_ok=True)

    attribute_path = os.path.join('../data_synthetic/DG_generated', f"synthetic_attributes_{dataset_name}.npy")
    feature_path = os.path.join('../data_synthetic/DG_generated', f"synthetic_features_{dataset_name}.npy")

    np.save(attribute_path, synthetic_attributes)
    np.save(feature_path, synthetic_features)

def main():
    parser = argparse.ArgumentParser(description="Training DoppelGANger...")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name for model to store in training_info (e.g. test_model, remember quotes)")
    parser.add_argument('--real_feature_path', type=str, required=True, help="Path to real_feature_path.npy (remember quotes)")
    parser.add_argument('--real_attribute_path', type=str, required=True, help="Path to real_attribute_path.npy (remember quotes)")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs for training.")
    
    args = parser.parse_args()
    
    train_generate(args.dataset_name, args.real_feature_path, args.real_attribute_path, args.epochs)

if __name__ == "__main__":
    main()