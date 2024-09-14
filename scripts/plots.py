import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pickle
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import random
import pandas as pd
import json
import re
from collections import defaultdict, Counter
from matplotlib.ticker import MaxNLocator
from typing import Callable, Dict, List, Optional, Tuple
from statsmodels.tsa.stattools import acf
from scripts.util import combine_vectors, define_attributes, create_sample_subset

def plot_autocorrelation(
    real_features_path: str, 
    real_attributes_path: str,
    syn_features_path: str,
    syn_attributes_path: str, 
    dataset_name: str
):
    '''
    Plot autocorrelation for three random examples. 

    Args: 
        real_features_path: Path to .npy file with real features.
        real_attributes_path: Path to .npy file with real attributes.
        syn_features_path: Path to .npy file with synthetic features.
        syn_attributes_path: Path to .npy file with synthetic attributes.
    '''
    real_features = np.load(real_features_path, allow_pickle=True)
    real_attributes = np.load(real_attributes_path, allow_pickle=True)
    synthetic_features = np.load(syn_features_path, allow_pickle=True)
    synthetic_attributes = np.load(syn_attributes_path, allow_pickle=True)

    real_data = combine_vectors(real_features, real_attributes)
    syn_data = combine_vectors(synthetic_features, synthetic_attributes)

    subjects, annotations = define_attributes(real_attributes)
    random_subjects = np.random.choice(subjects, size=3, replace=False)

    sns.set(style="darkgrid")

    plt.figure(figsize=(15, 10))

    for idx, subject in enumerate(random_subjects):
        for jdx, annotation in enumerate(annotations):
            real_subset = create_sample_subset(real_data, subject, annotation)
            syn_subset = create_sample_subset(syn_data, subject, annotation)

            real_sample = real_subset[np.random.randint(real_subset.shape[0]), :, :-2]
            syn_sample = syn_subset[np.random.randint(syn_subset.shape[0]), :, :-2]

            real_acf = acf(real_sample.flatten(), nlags=500)
            syn_acf = acf(syn_sample.flatten(), nlags=500)

            ax = plt.subplot(len(random_subjects), len(annotations), idx * len(annotations) + jdx + 1)
            
            ax.scatter(range(len(real_acf)), real_acf, alpha=0.7, label='Real', s=1)
            
            ax.scatter(range(len(syn_acf)), syn_acf, alpha=0.7, label='Synthetic', s=1)
            
            ax.legend()
            ax.set_title(f'Subject {subject}, {annotation}', fontsize=10)
            ax.set_xlabel('Lag', fontsize=10)
            ax.set_ylabel('ACF', fontsize=10)

    plt.suptitle(f'{dataset_name}', fontsize=16, ha='center')
    plt.tight_layout()
    plt.show()

def plot_distribution(
    real_features_path: str, 
    syn_features_path: str, 
    dataset_name: str):
    '''
    Plot distributions of real and synthetic features.
    
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


    sns.set(style="darkgrid")

    sns.kdeplot(prep_data, linewidth=2, label='Real') 
    sns.kdeplot(prep_data_hat, linewidth=2, linestyle='--', label='Synthetic', color='C1')

    plt.legend(loc='upper right', fontsize=10)

    plt.axvline(x=prep_data_hat.min(), color='C1', linestyle=':', linewidth=1)
    plt.axvline(x=prep_data_hat.max(), color='C1', linestyle=':', linewidth=1)

    plt.text(prep_data_hat.min(), plt.ylim()[1] * 0.6, 'Synthetic minimum', 
             color='C1', fontsize=8, ha='center', va='center')
    plt.text(prep_data_hat.max(), plt.ylim()[1] * 0.6, 'Synthetic maximum', 
             color='C1', fontsize=8, ha='center', va='center')

    plt.title(f'{dataset_name.capitalize()}', fontsize=12)

    plt.xlabel('μV', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    
    plt.legend()
    plt.show()

def plot_random_samples(
    real_features_path: str, 
    real_attributes_path: str,
    syn_features_path: str,
    syn_attributes_path: str,
    subject: str, 
    task: str, 
    dataset_name: str):
    '''
    Plot random samples from real and synthetic data for specific subject and task.

    Args: 
    - subject: str. A specific subject (e.g., "S4")
    - task: str. A specific task (either "left" or "right")
    '''

    real_features = np.load(real_features_path, allow_pickle=True)
    real_attributes = np.load(real_attributes_path, allow_pickle=True)
    synthetic_features = np.load(syn_features_path, allow_pickle=True)
    synthetic_attributes = np.load(syn_attributes_path, allow_pickle=True)

    real_data = combine_vectors(real_features, real_attributes)
    synthetic_data = combine_vectors(synthetic_features, real_attributes)

    def filter_indices(data, subject, task):
        return [i for i in range(len(data)) if data[i, 0, -2] == task and data[i, 0, -1] == subject]

    matching_real_indices = filter_indices(real_data, subject, task)
    matching_synthetic_indices = filter_indices(synthetic_data, subject, task)

    if matching_real_indices and matching_synthetic_indices:
        random_real_indices = random.sample(matching_real_indices, min(3, len(matching_real_indices)))
        random_synthetic_indices = random.sample(matching_synthetic_indices, min(3, len(matching_synthetic_indices)))

        time_steps = np.arange(real_features.shape[1]) 

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)

        for i, (real_idx, synthetic_idx) in enumerate(zip(random_real_indices, random_synthetic_indices)):
            selected_real_series = real_data[real_idx, :, :-2].astype(float)
            selected_synthetic_series = synthetic_data[synthetic_idx, :, :-2].astype(float)

            axes[i].plot(time_steps, selected_real_series[:, 0], label=f'Real (sample {real_idx})', c='blue')
            axes[i].plot(time_steps, selected_synthetic_series[:, 0], label=f'Synthetic (sample {synthetic_idx})', linestyle='--', c='orange')
            axes[i].set_ylabel('Channel Cz')
            axes[i].set_xlabel('Time steps')
            ticks = np.arange(0, len(time_steps), 100)
            ticks = np.append(ticks, 656)
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(ticks)
            axes[i].set_xlim(ticks[0], ticks[-1])
            axes[i].text(0.5, 1.15, f"{task.capitalize()} hand imagery", fontsize=18, ha='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{dataset_name.capitalize()}', fontsize=14)
            axes[i].legend()

        plt.tight_layout()
        plt.show()
    else:
        print(f"No matching samples found for subject {subject} and task {task}.")

def plot_confusion_matrices(
    folder_path: str, 
    condition: str,
    augment_percentage: Optional[int] = None):
    ''' 
    Plot confusion matrices from classification with or without augmentation
    from pickle files in the specified folder.

    Args:
        folder_path: str. Path to the folder containing classification results files.
        condition: whether plotting real data, synthetic data, high baseline or moderate baseline. 
            Takes either: "syn", "high_baseline", "moderate_baseline", or "real" (str)
        augment_percentage: Optional. Specify if you want to plot results from data augmentation conditions. 
            Takes either: 10, 50, or 100 (int). If not provided, default if results without augmentation.
    '''

    if augment_percentage is not None and augment_percentage not in [10, 50, 100]:
        raise ValueError("augment_percentage must be either 10, 50, or 100")

    if condition not in ["syn", "moderate_baseline", "high_baseline", "real"]:
        raise ValueError("condition must be either 'syn', 'moderate_baseline', 'high_baseline', or 'real'")

    if condition=="real" and augment_percentage is not None: 
        raise ValueError("No augmentation is possible with real data. Provide another condition or do not specify augment_percentage")

    if augment_percentage is not None:
        search_str = f"{augment_percentage}%"
        pickle_files = [f for f in os.listdir(folder_path) 
                        if f.endswith('.pkl') and search_str in f and condition in f]
    else:
        pickle_files = [f for f in os.listdir(folder_path) 
                        if f.endswith('.pkl') and '%' not in f and condition in f]

    num_files = len(pickle_files)


    if num_files == 0:
        print("No files found matching the criteria.")
        return

    num_cols = 3
    num_rows = (num_files + num_cols - 1) // num_cols  

    plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    sns.set(style="darkgrid")

    for idx, file in enumerate(pickle_files):
        classification_results = os.path.join(folder_path, file)
        dataset_name = os.path.splitext(os.path.basename(classification_results))[0]
        dataset_name = dataset_name[len("results_"):]
        suffix = '_dropout_0'
        if dataset_name.endswith(suffix):
            dataset_name = dataset_name[:-len(suffix)]
        dataset_name = dataset_name.replace('_', ' ')
        dataset_name = dataset_name.replace('sub', 'subjects,')
        dataset_name = dataset_name.replace('syn', 'synthetic')
        words = dataset_name.split(' ')
        if len(words) > 1:
            words.insert(1, 'data,')
        dataset_name = ' '.join(words)
        if '%' in dataset_name:
            percent_index = dataset_name.find('%')
            dataset_name = dataset_name[:percent_index + 1] + ' augmentation,' + dataset_name[percent_index + 1:]
        lines = dataset_name.split(' augmentation,')
        if len(lines) > 1:
            dataset_name = lines[0] + ' augmentation,\n' + lines[1]


        with open(classification_results, 'rb') as file:
            classification_results = pickle.load(file)

        confusion_matrix = classification_results['confusion_matrix']
        
        matrix_for_plot = np.zeros_like(confusion_matrix, dtype=float)
        
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                if i == j:
                    matrix_for_plot[i, j] = 2
                elif confusion_matrix[i, j] > 0:
                    matrix_for_plot[i, j] = 1

        total = np.sum(confusion_matrix)
        percentages = 100 * confusion_matrix / total

        annot = np.empty(confusion_matrix.shape, dtype=object)
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                count = confusion_matrix[i, j]
                percent = percentages[i, j]
                annot[i, j] = f"{count}\n({percent:.1f} %)"

        def custom_cmap():
            colors = [(0.5, 0.5, 0.5, 0.5), 
                    (1.0, 0.0, 0.0, 0.5), 
                    (0.0, 1.0, 0.0, 0.5)]
            cmap = mcolors.ListedColormap(colors)
            bounds = [-0.5, 0.5, 1.5, 2.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            return cmap, norm

        cmap, norm = custom_cmap()

        ax = plt.subplot(num_rows, num_cols, idx + 1)
        
        sns.heatmap(matrix_for_plot, annot=annot, fmt='', cmap=cmap, norm=norm,
                    xticklabels=['Left', 'Right'], 
                    yticklabels=['Left', 'Right'],
                    cbar=False, linewidths=0.5, linecolor='black', ax=ax,
                    annot_kws={"size": 22})

        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(f'{dataset_name.capitalize()}', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_evaluation_results(
    path_real_results: str, 
    path_baseline_moderate_results: str, 
    path_baseline_high_results: str, 
    path_synthetic_results: str, 
    dataset_name: str): 

    with open(path_real_results, 'r') as file:
        real_results = json.load(file)
    with open(path_baseline_moderate_results, 'r') as file:
        baseline_moderate_results = json.load(file)
    with open(path_baseline_high_results, 'r') as file:
        baseline_high_results = json.load(file)
    with open(path_synthetic_results, 'r') as file:
        synthetic_results = json.load(file)
    
    key_rename_map = {
        'global_ed_distance': 'Euclidean Distance (ED)',
        'global_mv_dtw_distance': 'Dependent MV-DTW Distance',
        'global_CCD_MSE': 'Cross-Correlation Difference (CCD)',
        #'global_ACF_MSE': 'Auto Correlation Difference (ACD)',
        #'global_ICD_percentage_error': 'Global ICD Percentage Error',
    }

    metrics_keys = [key for key in real_results['global_metrics'] if key in key_rename_map]

    x_labels = ['Real', 'BM', 'BH', 'Synthetic']
    dicts = [real_results, baseline_moderate_results, baseline_high_results, synthetic_results]

    sns.set(style="darkgrid")

    fig, axs = plt.subplots(1, len(metrics_keys), figsize=(len(metrics_keys) * 4, 6))

    colors = ['#FFA07A', '#FFA07A', '#FFA07A', '#3CB371']

    for i, key in enumerate(metrics_keys):
        values = [d['global_metrics'][key] for d in dicts]
        
        df = pd.DataFrame({
            'Configuration': x_labels,
            'Value': values
        })
        
        sns.barplot(data=df, x='Configuration', y='Value', ax=axs[i], hue='Configuration', palette=colors, dodge=False, legend=False)
        
        for p in axs[i].patches:
            axs[i].annotate(f'{p.get_height():.2f}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='bottom', fontsize=10, color='black')

        axs[i].set_title(key_rename_map.get(key, key), fontsize=12)
        axs[i].set_ylabel('Distance / Error', fontsize=10)
        axs[i].set_xlabel('')
        axs[i].set_ylim(0, max(values) * 1.2) 

    fig.suptitle(f'{dataset_name.capitalize()}', fontsize=16, ha='center')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_loss(
    path_loss_dict: str,
    dataset_name: str):
    ''' Plot loss for generator and discriminator. 
    '''
    with open(path_loss_dict, 'r') as file:
        data_dict = json.load(file)

    epochs = data_dict['losses']['epoch']
    discriminator_loss = data_dict['losses']['discriminator']
    generator_loss = data_dict['losses']['generator']

    df = pd.DataFrame({
        'Epoch': epochs,
        'Discriminator loss': discriminator_loss,
        'Generator loss': generator_loss
    })

    df_melted = df.melt(id_vars='Epoch', var_name='Loss Type', value_name='Loss')

    plt.figure(figsize=(12, 8))

    sns.set(style="darkgrid")

    sns.lineplot(
        data=df_melted,
        x='Epoch',
        y='Loss',
        hue='Loss Type',
        linewidth=0.8,
        palette={'Discriminator loss': 'orange', 'Generator loss': 'green'}
    )


    plt.title(f'{dataset_name.capitalize()}', fontsize=22)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title='Loss Type', fontsize=10)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_attributes(
    attribute_path: str,
    dataset_name: str
    ):

    attributes = np.load(attribute_path, allow_pickle=True)

    subject_ids = attributes[:, 1]
    annotations = attributes[:, 0]

    subject_annotation_counts = defaultdict(lambda: defaultdict(int))
    
    for subject, annotation in zip(subject_ids, annotations):
        subject_annotation_counts[subject][annotation] += 1
    
    sorted_subject_ids = sorted(subject_annotation_counts.keys())
    all_annotations = sorted(set(annotations))
    
    bar_width = 0.2
    index = np.arange(len(sorted_subject_ids))
    
    plt.figure(figsize=(12, 6))
    
    for i, annotation in enumerate(all_annotations):
        counts = [subject_annotation_counts[subject][annotation] for subject in sorted_subject_ids]
        plt.bar(index + i * bar_width, counts, bar_width, label=annotation)

    expected_task_samples = ((15 * 3) / 100 * 80) / 2
    plt.axhline(y=expected_task_samples, color='red', linestyle='--', label=f'Expected number of samples')
    
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Subject IDs')
    plt.ylabel('Count')
    plt.title(f'{dataset_name.capitalize()}')
    if len(sorted_subject_ids) < 26:
        plt.xticks(index + bar_width * (len(all_annotations) - 1) / 2, sorted_subject_ids)
    else:
        plt.xticks([])
    plt.legend()
    plt.show()

def plot_classification_results(
    folder_path: str, 
    to_excel_path: Optional[str] = None):
    ''' Create table with classification results. 

    Args: 
        folder_path: Path to folder containing pickle files
            with classification results. 
        to_excel: bool. Whether to store resulting table as Excel. 
    ''' 
    data = []
    filenames = []

    key_rename_map = {
        'accuracy': 'Accuracy %',
        'precision_left': 'Precision, left %',
        'precision_right': 'Precision, right %',
        'recall_left': 'Recall, left %',
        'recall_right': 'Recall, right %',
        'f1_left': 'F1, left',
        'f1_right': 'F1, right'
    }

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(folder_path, file_name)

            file_name = file_name[len("results_"):]
            suffix = '_dropout_0.pkl'
            if file_name.endswith(suffix):
                file_name = file_name[:-len(suffix)]
            else:
                file_name = file_name[:-4] 
            file_name = file_name.replace('_', ' ')
            file_name = file_name.replace('baseline', 'baseline data,')
            file_name = file_name.replace('sub', 'subjects,')
            if 'syn' in file_name: 
                file_name = file_name.replace('syn', 'synthetic data,')
            if 'real' in file_name: 
                file_name = file_name.replace('real', 'real data,')
            if '%' in file_name:
                percent_index = file_name.find('%')
                file_name = file_name[:percent_index + 1] + ' augmentation,' + file_name[percent_index + 1:]
            file_name = file_name.capitalize()

            with open(file_path, 'rb') as file:
                file_data = pickle.load(file)

            filtered_data = {key_rename_map[key]: value for key, value in file_data.items() if key in key_rename_map}

            data.append(filtered_data)
            filenames.append(file_name)
    
    df = pd.DataFrame(data, index=filenames)
    df = df.round(3)
    df['Accuracy %'] = df['Accuracy %'] * 100
    df['Precision, left %'] = df['Precision, left %'] * 100
    df['Precision, right %'] = df['Precision, right %'] * 100
    df['Recall, left %'] = df['Recall, left %'] * 100
    df['Recall, right %'] = df['Recall, right %'] * 100

    df = df.sort_index()

    if to_excel_path: 
        file_name = 'classification_results_table.xlsx'
        path = os.path.join(to_excel_path, file_name)
        df.to_excel(path, index=True)

    return df

def plot_hellinger(
    path_synthetic_eval_results_all_sub: str,
    path_synthetic_eval_results_25_sub_3: str,
    path_synthetic_eval_results_25_sub_15: str
):
    ''' Visualize the Hellinger distance for all synthetic datasets.
    '''

    with open(path_synthetic_eval_results_all_sub, 'r') as file:
        all_sub = json.load(file)
    with open(path_synthetic_eval_results_25_sub_3, 'r') as file:
        subset_3 = json.load(file)
    with open(path_synthetic_eval_results_25_sub_15, 'r') as file:
        subset_15 = json.load(file)
    
    key = 'hellinger_distance' 
    key_rename_map = {
        'hellinger_distance': 'Hellinger Distance'
    }

    values = [
        all_sub['global_metrics'].get(key, None),
        subset_3['global_metrics'].get(key, None),
        subset_15['global_metrics'].get(key, None)
    ]
    
    x_labels = ['3-ALL', '3-SUB', '15-SUB']

    df = pd.DataFrame({
        'Configuration': x_labels,
        'Value': values
    })
    
    sns.set(style="darkgrid")

    plt.figure(figsize=(7, 6))
    ax = sns.barplot(data=df, x='Configuration', y='Value', palette=['#ADD8E6', '#FFA07A', '#3CB371'])
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title(key_rename_map.get(key, key), fontsize=12)
    ax.set_ylabel('Distance, percentage', fontsize=10)
    ax.set_xlabel('')
    ax.set_ylim(0, max(values) * 1.2) 

    plt.suptitle('Hellinger distances for all synthetic datasets', fontsize=16, ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_icd(
    path_real_results: str, 
    path_baseline_moderate_results: str, 
    path_baseline_high_results: str, 
    path_synthetic_results: str, 
    dataset_name: str
):
    ''' Visualize the ICD percentage error for different datasets.
    '''

    with open(path_real_results, 'r') as file:
        real_results = json.load(file)
    with open(path_baseline_moderate_results, 'r') as file:
        baseline_moderate_results = json.load(file)
    with open(path_baseline_high_results, 'r') as file:
        baseline_high_results = json.load(file)
    with open(path_synthetic_results, 'r') as file:
        synthetic_results = json.load(file)
    
    key = 'global_ICD_percentage_error' 
    key_rename_map = {
        'global_ICD_percentage_error': 'Intra-Class Distance (ICD)',
    }

    values = [
        real_results['global_metrics'].get(key, None),
        baseline_moderate_results['global_metrics'].get(key, None),
        baseline_high_results['global_metrics'].get(key, None),
        synthetic_results['global_metrics'].get(key, None)
    ]
    
    x_labels = ['Real', 'BM', 'BH', 'Synthetic']

    df = pd.DataFrame({
        'Configuration': x_labels,
        'Value': values
    })
    
    sns.set(style="darkgrid")

    plt.figure(figsize=(5, 6))
    ax = sns.barplot(data=df, x='Configuration', y='Value', palette=['#FFA07A', '#FFA07A', '#FFA07A', '#3CB371'])

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title(key_rename_map.get(key, key), fontsize=12)
    ax.set_ylabel('Percentage error', fontsize=10)
    ax.set_xlabel('')
    ax.set_ylim(0, max(values) * 1.2) 

    plt.suptitle(f'{dataset_name}', fontsize=16, ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_innd(): 
    real_mean_ed_3_all = 1607.5505360245572
    real_mean_ed_3_sub = 1803.7172006359674
    real_mean_ed_15_sub = 141.9029722767518

    innd_3_all = 334.61395596139107
    innd_3_sub = 378.5518743105074
    innd_15_sub = 84.26961700030687

    perc_diff_3_all = (innd_3_all / real_mean_ed_3_all) * 100
    perc_diff_3_sub = (innd_3_sub / real_mean_ed_3_sub) * 100 
    perc_diff_15_sub = (innd_15_sub / real_mean_ed_15_sub) * 100

    values = [perc_diff_3_all, perc_diff_3_sub, perc_diff_15_sub]
    x_labels = ['3-ALL', '3-SUB', '15-SUB']

    df = pd.DataFrame({
        'Configuration': x_labels,
        'Value': values
    })

    sns.set(style="darkgrid")

    plt.figure(figsize=(5, 6))
    ax = sns.barplot(data=df, x='Configuration', y='Value', palette=['#ADD8E6', '#FFA07A', '#3CB371'])

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black')

    ax.set_title('INND', fontsize=14)
    ax.set_ylabel('Percentage of real mean ED', fontsize=10)
    ax.set_xlabel('')
    ax.set_ylim(0, max(values) * 1.2) 

    plt.tight_layout()
    plt.show()


def plot_tsne_tasks(
    real_features_path: str, 
    real_attributes_path: str, 
    syn_features_path: str, 
    syn_attributes_path: str, 
    dataset_name: str):
    ''' 
    Compute and plot t-SNE with marked task attributes.
    ''' 
    real_features = np.load(real_features_path, allow_pickle=True)
    real_attributes = np.load(real_attributes_path, allow_pickle=True)
    syn_features = np.load(syn_features_path, allow_pickle=True)
    syn_attributes = np.load(syn_attributes_path, allow_pickle=True)

    real_labels_task = real_attributes[:, 0]
    synthetic_labels_task = syn_attributes[:, 0]

    combined_features = np.concatenate((real_features, syn_features), axis=0)
    combined_labels_task = np.concatenate((real_labels_task, synthetic_labels_task), axis=0)

    combined_features_mean = np.mean(combined_features, axis=1)

    number_of_samples = real_features.shape[0]
    colors = ["C0"] * number_of_samples + ["C1"] * number_of_samples

    shape_map = {"left": 'o', "right": '^'}
    shapes = [shape_map[label] for label in combined_labels_task]

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(combined_features_mean)

    fig, ax = plt.subplots()

    for i in range(number_of_samples):
        ax.scatter(tsne_results[i, 0], tsne_results[i, 1], 
                   c="C0", alpha=0.3, s=30, 
                   marker=shapes[i], label="Real" if i == 0 else "", edgecolor='w')

    for i in range(number_of_samples, tsne_results.shape[0]):
        ax.scatter(tsne_results[i, 0], tsne_results[i, 1], 
                   c="C1", alpha=0.3, s=30, 
                   marker=shapes[i], label="Synthetic" if i == number_of_samples else "", edgecolor='w')

    shape_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=7, label='left'),
                    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=7, label='right')]

    color_legend = [plt.Line2D([0], [0], color='C0', lw=2, label='Real'),
                    plt.Line2D([0], [0], color='C1', lw=2, label='Synthetic')]

    ax.legend(handles=shape_legend + color_legend, loc='best', fontsize=7)

    ax.legend(handles=shape_legend + color_legend, loc='best')
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f't-SNE plot for {dataset_name}', fontsize=12)

    plt.show()

def plot_tsne_subjects(
    real_features_path: str, 
    real_attributes_path: str, 
    syn_features_path: str, 
    syn_attributes_path: str, 
    dataset_name: str):
    ''' 
    Compute and plot t-SNE with marked subject attributes.
    ''' 
    real_features = np.load(real_features_path, allow_pickle=True)
    real_attributes = np.load(real_attributes_path, allow_pickle=True)
    syn_features = np.load(syn_features_path, allow_pickle=True)
    syn_attributes = np.load(syn_attributes_path, allow_pickle=True)

    number_of_samples = len(real_features)

    real_labels_subject = real_attributes[:, 1]
    synthetic_labels_subject = syn_attributes[:, 1]

    combined_features = np.concatenate((real_features, syn_features), axis=0)
    combined_labels_subject = np.concatenate((real_labels_subject, synthetic_labels_subject), axis=0)
    combined_features_mean = np.mean(combined_features, axis=1)

    label_encoder = LabelEncoder()
    numeric_labels_subject = label_encoder.fit_transform(combined_labels_subject)

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(combined_features_mean)

    unique_labels = np.unique(numeric_labels_subject)
    palette = sns.color_palette("tab10", len(unique_labels))

    colors = [palette[label] for label in numeric_labels_subject]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].scatter(tsne_results[:number_of_samples, 0], tsne_results[:number_of_samples, 1], 
                    c=colors[:number_of_samples], alpha=0.5, label="Real", s=15)
    axes[0].set_title(f't-SNE plot for real data ({dataset_name})', fontsize=12)
    axes[0].grid(False)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(tsne_results[number_of_samples:, 0], tsne_results[number_of_samples:, 1], 
                    c=colors[number_of_samples:], alpha=0.5, label="Synthetic", s=15)
    axes[1].set_title(f't-SNE plot for synthetic data ({dataset_name})', fontsize=12)
    axes[1].grid(False)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    handles = [plt.Line2D([0], [0], marker='o', color=palette[i], linestyle='', markersize=6) 
               for i in range(len(unique_labels))]
    fig.legend(handles, label_encoder.inverse_transform(unique_labels), title="Subject", 
               bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.tight_layout()
    plt.show()
