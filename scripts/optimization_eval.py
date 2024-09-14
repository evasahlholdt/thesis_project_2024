import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Any
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_optimization(optimization_results: Dict[str, Any], number_of_epochs: Optional[int] = 10, n_candidates: Optional[int] = 5
) -> Dict[str, Any]:
    ''' 
    Apply methods for evaluating best hyperparameter configuration for DG.
    ''' 
    survivors_dict = _remove_early_stopping_instances(optimization_results)
    mean_dtw_dict = _compute_mean_dtw(survivors_dict, number_of_epochs)
    mean_attribute_error_dict = _compute_mean_attribute_percentage(mean_dtw_dict, number_of_epochs)
    best_candidates_dict = _get_lowest_mean_dtw(mean_attribute_error_dict, n_candidates)
    _print_metrics(best_candidates_dict)

    return best_candidates_dict


def _remove_early_stopping_instances(optimization_results: Dict[str, Any]
) -> Dict[str, Any]:
    ''' 
    Remove all instances which did not pass early stopping
    '''
    survivors_dict = {
        key: value for key, value in optimization_results.items()
        if len(value['losses']['epoch']) >= 11 and len(value['metrics']['epoch']) >= 11
    }
    return survivors_dict

def _compute_mean_dtw(survivors_dict: Dict[str, Any], number_of_epochs: Optional[int] = 10
) -> Dict[str, Any]:
    ''' 
    Compute the mean DTW for a specified number of last epochs. 

    Args: 
    - 
    - number_of_epochs: int. Number of last epochs to average over. Default: 10.
    '''
    mean_dtw_dict = {}

    for key, value in survivors_dict.items():
        mean_dtw_values = value.get('metrics', {}).get('mean_dtw', [])
        
        mean_dtw_result = np.mean(mean_dtw_values[-number_of_epochs:])
        
        mean_dtw_dict[key] = value.copy() 
        mean_dtw_dict[key]['mean_dtw_result'] = mean_dtw_result
    
    return mean_dtw_dict

def _compute_mean_attribute_percentage(mean_dtw_dict: Dict[str, Any], number_of_epochs: Optional[int] = 10
) -> Dict[str, Any]:
    '''
    Compute the mean percentage error on attribute distribution

    Args: 
    - 
    - number_of_epochs: int. Number of last epochs to average over. Default: 10.
    '''
    mean_attribute_error_dict = {}

    for key, value in mean_dtw_dict.items():
        average_attribute_error_values = value.get('metrics', {}).get('average_attribute_error', [])
        
        mean_attribute_error = np.mean(average_attribute_error_values[-number_of_epochs:])
        
        mean_attribute_error_dict[key] = value.copy() 
        mean_attribute_error_dict[key]['mean_attribute_error'] = mean_attribute_error
    
    return mean_attribute_error_dict

def _get_lowest_mean_dtw(mean_attribute_error_dict: Dict[str, Any], n_candidates: Optional[int] = 5
) -> Dict[str, Any]:
    '''
    Get the items with the lowest mean MV-DTW distance. 

    Args: 
    - mean_attribute_error_dict: Dict. 
    - n: int (Optional). Number of best candidates. Default = 5
    '''
    best_candidates = sorted(mean_attribute_error_dict.items(), key=lambda item: item[1].get('mean_dtw_result', float('inf')))[:n_candidates]
    
    best_candidates_dict = {key: value for key, value in best_candidates}
    
    return best_candidates_dict

def _print_metrics(best_candidates_dict: Dict[str, Any]):
    ''' 
    Print results.
    ''' 
    for key, value in best_candidates_dict.items():
        mean_dtw_last = value.get('mean_dtw_result', None)
        mean_attribute_error_values = value['metrics'].get('mean_attribute_error', [])
        
        if mean_attribute_error_values:
            mean_attribute_error = mean_attribute_error_values[-1]
        else:
            mean_attribute_error = None
        
        params = value.get('params', {})

        print(f"Key: {key}")
        print(f"Mean DTW: {mean_dtw_last}")
        print(f"Mean attribute error, %: {mean_attribute_error}")
        print(f"Params: {params}\n")

def plot_discriminator_loss(best_candidates):
    plt.figure(figsize=(12, 8))

    sns.set(style="darkgrid")

    for key, value in best_candidates.items():
        epochs = value['losses']['epoch']
        discriminator_values = value['losses']['discriminator']

        df = pd.DataFrame({'Epoch': epochs, 'Discriminator loss': discriminator_values})

        sns.lineplot(data=df, x='Epoch', y='Discriminator loss', label=f'Configuration: {key}', marker='o', linewidth=2)

    plt.title('Discriminator loss over epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Discriminator loss', fontsize=14)
    plt.legend(title='Configurations', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_discriminator_loss_for_key(best_candidates, specific_key):
    plt.figure(figsize=(12, 8))

    if specific_key in best_candidates:
        value = best_candidates[specific_key]
        epochs = value['losses']['epoch']
        discriminator_values = value['losses']['discriminator']
        
        df = pd.DataFrame({'Epoch': epochs, 'Discriminator loss': discriminator_values})

        sns.set(style="darkgrid")

        sns.lineplot(data=df, x='Epoch', y='Discriminator loss', marker='o', linewidth=2, color='black')

        plt.title(f'Discriminator loss over epochs for configuration {specific_key}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Discriminator loss', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print(f"Key {specific_key} not found in the provided dictionary.")

def plot_generator_loss(best_candidates):
    plt.figure(figsize=(12, 8))

    for key, value in best_candidates.items():
        epochs = value['losses']['epoch']
        generator_values = value['losses']['generator']
        
        df = pd.DataFrame({'Epoch': epochs, 'Generator loss': generator_values})

        sns.set(style="darkgrid")

        sns.lineplot(data=df, x='Epoch', y='Generator loss', label=f'Configuration: {key}', marker='o', linewidth=2)

    plt.title('Generator loss over epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Generator loss', fontsize=14)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_generator_loss_for_key(best_candidates, specific_key):
    plt.figure(figsize=(12, 8))

    if specific_key in best_candidates:
        value = best_candidates[specific_key]
        epochs = value['losses']['epoch']
        generator_values = value['losses']['generator']

        df = pd.DataFrame({'Epoch': epochs, 'Generator loss': generator_values})

        sns.set(style="darkgrid")

        sns.lineplot(data=df, x='Epoch', y='Generator loss', marker='o', linewidth=2, color='black')

        plt.title(f'Generator loss over epochs for configuration {specific_key}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Generator loss', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print(f"Key {specific_key} not found in the provided dictionary.")

def plot_mean_dtw(best_candidates):
    plt.figure(figsize=(12, 8))

    sns.set(style="darkgrid")

    for key, value in best_candidates.items():
        epochs = value['metrics']['epoch']
        dtw_values = value['metrics']['mean_dtw']
        
        df = pd.DataFrame({'Epoch': epochs, 'Mean MV-DTW': dtw_values})

        sns.lineplot(data=df, x='Epoch', y='Mean MV-DTW', label=f'Configuration: {key}', marker='o')

    plt.title('Mean MV-DTW over epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean MV-DTW', fontsize=14)
    plt.legend(title='Configurations', fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_mean_attribute_error(best_candidates):
    plt.figure(figsize=(12, 8))

    sns.set(style="darkgrid")

    for key, value in best_candidates.items():
        epochs = value['metrics']['epoch']
        percentage_values = value['metrics']['average_attribute_error']
        
        df = pd.DataFrame({'Epoch': epochs, 'Mean attribute error (%)': percentage_values})

        sns.lineplot(data=df, x='Epoch', y='Mean attribute error (%)', label=f'Configuration: {key}', marker='o')

    plt.title('Mean attribute error (%) over epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean attribute error (%)', fontsize=14)
    plt.legend(title='Configurations', fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_attribute_error_for_key(best_candidates, specific_key):
    plt.figure(figsize=(12, 8))

    sns.set(style="darkgrid")

    if specific_key in best_candidates:
        value = best_candidates[specific_key]

        epochs = value['metrics']['epoch']
        percentage_values = value['metrics']['average_attribute_error']
        
        df = pd.DataFrame({'Epoch': epochs, 'Mean attribute error (%)': percentage_values})

        sns.lineplot(data=df, x='Epoch', y='Mean attribute error (%)', marker='o', linewidth=2, color='black')

        plt.title(f'Mean attribute error (%) over epochs for configuration {specific_key}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Mean attribute error (%)', fontsize=14)
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print(f"Key {specific_key} not found in the provided dictionary.")

def plot_mean_dtw_for_key(data_dict, specific_key):
    plt.figure(figsize=(12, 6))

    # Check if the specific key exists in the dictionary
    if specific_key in data_dict:
        value = data_dict[specific_key]

        epochs = value['metrics']['epoch']
        dtw_values = value['metrics']['mean_dtw']

        # Create a DataFrame for Seaborn
        df = pd.DataFrame({'Epoch': epochs, 'Mean MV-DTW': dtw_values})

        # Set Seaborn style
        sns.set(style="darkgrid")

        # Create the line plot with a different color
        sns.lineplot(data=df, x='Epoch', y='Mean MV-DTW', marker='o', color='black', linewidth=2)

        plt.title(f'Mean MV-DTW over epochs for configuration {specific_key}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Mean MV-DTW', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Key {specific_key} not found in the provided dictionary.")