import numpy as np
import os
import json
from typing import Optional
from dtaidistance.dtw_ndim import distance as multi_dtw_distance
import statsmodels.stats.contingency_tables as sm

from scripts.metrics import euclidean_distance, feature_correlation, auto_correlation, hellinger_distance 
from scripts.util import combine_vectors, define_attributes, create_sample_subset

class Evaluate:
    """
    Evaluate synthetic dataset using diverse metrics.

    Args: 
        - real_feature_vector: np.ndarray
        - real_attribute_vector: np.ndarray
        - syn_feature_vector: np.ndarray
        - syn_attribute_vector: np.ndarray
        - dataset_name: str. Common name for datasets being compared (e.g. samples_channel_subset)
        - result_folder_path: Optional[str]. Path to save evaluation results. Default is None.
    """
    def __init__(
        self,
        real_features_path: str,
        real_attributes_path: str,
        syn_features_path: str,
        syn_attributes_path: str,
        dataset_name: str,
        result_folder_path: str
    ):
        
        self.real_feature_vector = np.load(real_features_path, allow_pickle=True)
        self.real_attribute_vector = np.load(real_attributes_path, allow_pickle=True)
        self.syn_feature_vector = np.load(syn_features_path, allow_pickle=True)
        self.syn_attribute_vector = np.load(syn_attributes_path, allow_pickle=True)
        self.dataset_name = dataset_name

        self.real_data = combine_vectors(self.real_feature_vector, self.real_attribute_vector)
        self.syn_data = combine_vectors(self.syn_feature_vector, self.syn_attribute_vector)

        self.result_folder_path = result_folder_path

        self.ed_distances = None
        self.mv_dtw_distances = None
        self.feature_correlations = None
        self.autocorrelation_distances = None
        self.intra_class_distances = None
        self.hellinger_distance = None
        self.onnd = None
        self.innd = None

        self.global_metrics = {}

    def evaluate(self,
        use_euclidean: Optional[bool] = False,
        use_mv_dtw: Optional[bool] = False,
        use_feature_correlations: Optional[bool] = False,
        use_autocorrelation: Optional[bool] = False,
        use_icd: Optional[bool] = False,
        use_hellinger_distance: Optional[bool] = False,
        use_onnd: Optional[bool] = False,
        use_innd: Optional[bool] = False
    ): 
        ''' 
        Main function to run full evaluation.
        ''' 
        if use_euclidean:
            if self.ed_distances is not None:
                print("ED already computed")
            else: 
                print("Computing Euclidean distances...")
                self._compute_ed()
                print("Finished computing Euclidean distances")
        if use_mv_dtw:
            if self.mv_dtw_distances is not None:
                print("MV-DTW already computed")
            else: 
                print("Computing MV-DTW distances...")
                self._compute_mv_dtw()
                print("Finished computing MV-DTW distances")
        if use_feature_correlations:
            if self.feature_correlations is not None:
                print("Feature correlations already computed")
            else: 
                print("Computing feature correlations...")
                self._compute_feature_correlations()
                print("Finished computing feature correlations")
        if use_autocorrelation:
            if self.autocorrelation_distances is not None:
                print("Autocorrelation distances already computed")
            else: 
                print("Computing autocorrelation distances...")
                self._compute_autocorrelation()
                print("Finished computing autocorrelation distances")
        if use_icd:
            if self.intra_class_distances is not None:
                print("Autocorrelation distances already computed")
            else: 
                print("Computing intra class distances...")
                self._compute_icd()
                print("Finished computing intra class distances")
        if use_hellinger_distance:
            if self.hellinger_distance is not None:
                print("Hellinger distance already computed")
            else:     
                print("Computing Hellinger distance...")
                self._compute_hellinger_distance()
                print("Finished computing Hellinger distance")
        if use_onnd:
            if self.onnd is not None:
                print("ONND already computed")
            else:    
                print("Computing ONND...")
                self.onnd = self._compute_ONND()
                print("Finished computing ONND")
        if use_innd:
            if self.innd is not None:
                print("INND already computed")
            else:    
                print("Computing INND...")
                self.innd = self._compute_INND()
                print("Finished computing INND")
        self._save_evaluation_results()

    def _compute_ed(self):
        """
        Compute Euclidean distances between real and synthetic samples for each subject and annotation,
        and calculate the mean Euclidean distance for each subject-annotation pair as well as the overall mean.
        """
        subjects, annotations = define_attributes(self.real_attribute_vector)
        ed_distances = {}
        all_distances = []

        for subject in subjects: 
            for annotation in annotations:
                real_subset = create_sample_subset(self.real_data, subject, annotation)
                syn_subset = create_sample_subset(self.syn_data, subject, annotation)
                
                if syn_subset.shape[0] == 0:
                    ed_distances[(subject, annotation)] = {
                        'n_synthetic_samples': 0,
                        'n_real_samples': real_subset.shape[0], 
                        'ed_distance': np.nan,
                    }
                    continue

                real_samples = real_subset[:, :, :-2]
                syn_samples = syn_subset[:, :, :-2]

                distances = [
                    euclidean_distance(i_real, j_syn)
                    for i_real in real_samples
                    for j_syn in syn_samples
                ]

                mean_distance = np.mean(distances)
                all_distances.extend(distances) 

                ed_distances[(subject, annotation)] = {
                    'n_synthetic_samples': syn_subset.shape[0],
                    'n_real_samples': real_subset.shape[0], 
                    'ed_distance': mean_distance
                }
        
        self.global_metrics.update({
            'global_ed_distance': np.mean(all_distances) if all_distances else np.nan})
        self.ed_distances = ed_distances

    def _compute_mv_dtw(self):
        """
        Compute MV-DTW distances between real and synthetic samples for each subject and annotation,
        and calculate the mean MV-DTW distance.
        """
        subjects, annotations = define_attributes(self.real_attribute_vector)
        mv_dtw_distances = {}
        all_distances = []

        for subject in subjects: 
            for annotation in annotations:
                real_subset = create_sample_subset(self.real_data, subject, annotation)
                syn_subset = create_sample_subset(self.syn_data, subject, annotation)
                
                if syn_subset.shape[0] == 0:
                    mv_dtw_distances[(subject, annotation)] = {
                        'n_synthetic_samples': 0,
                        'n_real_samples': real_subset.shape[0], 
                        'mv_dtw_distance': np.nan,
                    }
                    continue

                real_samples = real_subset[:, :, :-2]
                syn_samples = syn_subset[:, :, :-2]

                distances = [
                    multi_dtw_distance(i_real.astype(np.double), j_syn.astype(np.double), use_c=True)
                    for i_real in real_samples
                    for j_syn in syn_samples
                ]
                
                mean_distance = np.mean(distances)
                all_distances.extend(distances)

                mv_dtw_distances[(subject, annotation)] = {
                    'n_synthetic_samples': syn_subset.shape[0],
                    'n_real_samples': real_subset.shape[0], 
                    'mv_dtw_distance': mean_distance
                }

        self.global_metrics.update({
            'global_mv_dtw_distance': np.mean(all_distances) if all_distances else np.nan})
        self.mv_dtw_distances = mv_dtw_distances

    def _compute_feature_correlations(self):
        ''' 
        Compute feature correlations using PCC.
        '''
        subjects, annotations = define_attributes(self.real_attribute_vector)
        feature_correlations = {}

        for subject in subjects: 
            for annotation in annotations:
                real_subset = create_sample_subset(self.real_data, subject, annotation)
                syn_subset = create_sample_subset(self.syn_data, subject, annotation)
                
                if syn_subset.shape[0] == 0:
                    feature_correlations[(subject, annotation)] = {
                        'n_synthetic_samples': 0,
                        'n_real_samples': real_subset.shape[0],
                        'CCD': np.nan,
                        'CCD_mean_difference': np.nan,
                        'CCD_MSE': np.nan
                    }
                    continue

                correlation_matrices, CCD_mean_difference, CCD_MSE = feature_correlation(real_subset, syn_subset)

                feature_correlations[(subject, annotation)] = {
                    'n_synthetic_samples': syn_subset.shape[0],
                    'n_real_samples': real_subset.shape[0],
                    'CCD': correlation_matrices,
                    'CCD_mean_difference': CCD_mean_difference,
                    'CCD_MSE': CCD_MSE
                }

        self.global_metrics.update({
            'global_CCD_mean_difference': np.nanmean(
                [metrics['CCD_mean_difference'] for metrics in feature_correlations.values()]
            ),
            'global_CCD_MSE': np.nanmean(
                [metrics['CCD_MSE'] for metrics in feature_correlations.values()]
            )
        })

        self.feature_correlations = feature_correlations

    def _compute_autocorrelation(self):
        """
        Compute Autocorrelation distance (ACD) and global metrics.
        """
        subjects, annotations = define_attributes(self.real_attribute_vector)
        autocorrelation_distances = {}

        for subject in subjects: 
            for annotation in annotations:
                real_subset = create_sample_subset(self.real_data, subject, annotation)
                syn_subset = create_sample_subset(self.syn_data, subject, annotation)
                
                if syn_subset.shape[0] == 0:
                    autocorrelation_distances[(subject, annotation)] = {
                        'n_synthetic_samples': 0,
                        'n_real_samples': real_subset.shape[0],
                        'ACF_syn': np.nan,
                        'ACF_real': np.nan,
                        'ACF_MSE': np.nan
                    }
                    continue

                acf_syn = np.array([auto_correlation(i_syn) for i_syn in syn_subset])
                acf_real = np.array([auto_correlation(i_real) for i_real in real_subset])

                mean_acf_syn = np.mean(acf_syn, axis=0)
                mean_acf_real = np.mean(acf_real, axis=0)
                acf_mse = np.mean((mean_acf_real - mean_acf_syn) ** 2)

                autocorrelation_distances[(subject, annotation)] = {
                    'n_synthetic_samples': syn_subset.shape[0],
                    'n_real_samples': real_subset.shape[0], 
                    'ACF_syn': mean_acf_syn,
                    'ACF_real': mean_acf_real,
                    'ACF_MSE': acf_mse
                }

        acf_mse_values = [metrics['ACF_MSE'] for metrics in autocorrelation_distances.values() if metrics['ACF_MSE'] is not None]

        global_acf_mse = np.nanmean(acf_mse_values) if acf_mse_values else None

        self.global_metrics.update({
            'global_ACF_MSE': global_acf_mse if global_acf_mse is not None else np.nan
        })

        self.autocorrelation_distances = autocorrelation_distances


    def _compute_icd(self):
        ''' 
        Compute intra class distance (ICD) with Euclidean distance.
        '''
        subjects, annotations = define_attributes(self.real_attribute_vector)
        intra_class_distances = {}

        for subject in subjects: 
            for annotation in annotations:
                real_subset = create_sample_subset(self.real_data, subject, annotation)
                syn_subset = create_sample_subset(self.syn_data, subject, annotation)
                
                if syn_subset.shape[0] <= 1: 
                    intra_class_distances[(subject, annotation)] = {
                        'n_synthetic_samples': syn_subset.shape[0],
                        'n_real_samples': real_subset.shape[0],
                        'ICD_syn': np.nan,
                        'ICD_real': np.nan,
                        'ICD_percentage_error': np.nan
                    }
                    continue

                icd_syn = [
                    euclidean_distance(syn_subset[i, :, :], syn_subset[k, :, :])
                    for i in range(syn_subset.shape[0])
                    for k in range(syn_subset.shape[0])
                    if k != i
                ]

                icd_real = [
                    euclidean_distance(real_subset[i, :, :], real_subset[k, :, :])
                    for i in range(real_subset.shape[0])
                    for k in range(real_subset.shape[0])
                    if k != i
                ]

                mean_icd_syn = np.nanmean(icd_syn)
                mean_icd_real = np.nanmean(icd_real)

                intra_class_distances[(subject, annotation)] = {
                    'n_synthetic_samples': syn_subset.shape[0],
                    'n_real_samples': real_subset.shape[0], 
                    'ICD_syn': mean_icd_syn,
                    'ICD_real': mean_icd_real,
                    'ICD_percentage_error': ((mean_icd_syn - mean_icd_real) / mean_icd_real * 100)
                }

        self.global_metrics.update({
            'global_ICD_syn': np.nanmean([metrics['ICD_syn'] for metrics in intra_class_distances.values()]),
            'global_ICD_real': np.nanmean([metrics['ICD_real'] for metrics in intra_class_distances.values()]),
            'global_ICD_percentage_error': (
                (
                    np.nanmean([metrics['ICD_syn'] for metrics in intra_class_distances.values()]) -
                    np.nanmean([metrics['ICD_real'] for metrics in intra_class_distances.values()])
                ) / np.nanmean([metrics['ICD_real'] for metrics in intra_class_distances.values()]) * 100
                if not np.isnan(np.nanmean([metrics['ICD_real'] for metrics in intra_class_distances.values()])) else np.nan
            )
        })


        self.intra_class_distances = intra_class_distances

    def _compute_ONND(self):
        ''' 
        Compute ONND.
        '''

        onnd_scores = []

        counter = 0

        for real_sample in self.real_data:
            # Dropping MV-DTW as it is too heavy
            # distances = np.array([multi_dtw_distance(real_sample[:, :-2].astype(np.double), syn_sample[:, :-2].astype(np.double), use_c=True) for syn_sample in self.syn_data])
            distances = np.array([euclidean_distance(real_sample[:, :-2], syn_sample[:, :-2]) for syn_sample in self.syn_data])
            onnd = np.min(distances)
            onnd_scores.append(onnd)

            counter += 1

            if counter % 50 == 0:
                print(f"Processed {counter} of {self.real_data.shape[0]} real samples")

        global_onnd = np.mean(onnd_scores)

        self.global_metrics.update({
            'global_onnd': global_onnd
        })

        self.onnd = global_onnd
    
    def _compute_INND(self):
        ''' 
        Compute INND.

        '''
        innd_scores = []

        counter = 0 

        for syn_sample in self.syn_data:
            distances = np.array([euclidean_distance(real_sample[:, :-2], syn_sample[:, :-2]) for real_sample in self.real_data])
            innd = np.min(distances)
            innd_scores.append(innd)

            counter += 1

            if counter % 50 == 0:
                print(f"Processed {counter} of {self.syn_data.shape[0]} synthetic samples")


        global_innd = np.mean(innd_scores)
    
        self.global_metrics.update({
            'global_innd': global_innd
        })

        self.innd = global_innd

    def _compute_hellinger_distance(self):
        '''
        Compute Hellinger distance.
        '''
        hellinger_distance_val = hellinger_distance(self.real_attribute_vector, self.syn_attribute_vector)
        self.hellinger_distance = hellinger_distance_val
        self.global_metrics['hellinger_distance'] = hellinger_distance_val

    def _save_evaluation_results(self):
        ''' 
        Save results from evaluation. 
        ''' 

        if self.result_folder_path: 
            os.makedirs(self.result_folder_path, exist_ok=True)
            evaluation_results = {
            'global_metrics': self.global_metrics
            }
            filepath = os.path.join(self.result_folder_path, f"evaluation_results_{self.dataset_name}.json")
            try:
                with open(filepath, 'w') as f:
                    json.dump(evaluation_results, f, indent=4)
                print(f"Evaluation results stored as evaluation_results_{self.dataset_name}.json at location {self.result_folder_path}")
            except IOError as e:
                print(f"Error saving results: {e}")
        else: 
            os.makedirs('../results/results_evaluation', exist_ok=True)

            evaluation_results = {
                'global_metrics': self.global_metrics
            }
            
            filepath = os.path.join('../results/results_evaluation', f"evaluation_results_{self.dataset_name}.json")

            try:
                with open(filepath, 'w') as f:
                    json.dump(evaluation_results, f, indent=4)
                print(f"Evaluation results stored as evaluation_results_{self.dataset_name}.json in folder results/results_evaluation")
            except IOError as e:
                print(f"Error saving results: {e}")


