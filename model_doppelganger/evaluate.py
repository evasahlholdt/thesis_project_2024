import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dtaidistance.dtw_ndim import distance as multi_dtw_distance
from collections import Counter

class EvaluateTraining:
    """
    Evaluate synthetic dataset using MV-DTW and Hellinger distance during training.

    Args: 
        - real_feature_vector: np.ndarray
        - real_attribute_vector: np.ndarray
        - syn_feature_vector: np.ndarray
        - syn_attribute_vector: np.ndarray
    """
    def __init__(
        self,
        real_feature_vector: np.ndarray, 
        real_attribute_vector: np.ndarray, 
        syn_feature_vector: np.ndarray, 
        syn_attribute_vector: np.ndarray
    ):

        self.real_feature_vector = real_feature_vector
        self.real_attribute_vector = real_attribute_vector
        self.syn_feature_vector = syn_feature_vector
        self.syn_attribute_vector = syn_attribute_vector

        self.real_data = self._combine_vectors(self.real_feature_vector, self.real_attribute_vector)
        self.syn_data = self._combine_vectors(self.syn_feature_vector, self.syn_attribute_vector)

        self.local_metrics = None
        self.global_metrics = None

    def evaluate(self):
        ''' 
        Compute dependent multivariate dynamic time warping (MV-DTW)
        and attribute distribution error. 
        '''

        subjects, annotations = self._define_attributes(self.real_attribute_vector) #, self.syn_attribute_vector
        
        local_metrics = {}
        global_metrics = {}

        for subject in subjects:
            for annotation in annotations:

                real_subset = self._create_sample_subset(self.real_data, subject, annotation)
                syn_subset = self._create_sample_subset(self.syn_data, subject, annotation)
                
                if syn_subset.shape[0] == 0:
                    local_metrics[(subject, annotation)] = {
                        'mean_dtw': np.nan, 
                        'n_synthetic_samples': 0,
                        'n_real_samples': real_subset.shape[0],
                        'attribute_distribution_percentage_error': ((0 - real_subset.shape[0]) / real_subset.shape[0]) * 100
                    }
                    
                else:

                    temp_dtw = []
                    
                    for i in range(syn_subset.shape[0]):

                        i_syn = syn_subset[i, :, :]

                        for j in range(real_subset.shape[0]):
                            
                            j_real = real_subset[j, :, :]

                            temp_dtw.append(multi_dtw_distance(j_real.astype(np.double), i_syn.astype(np.double), use_c=True))

                    local_metrics[(subject, annotation)] = {
                        'mean_dtw': np.mean(temp_dtw).item(),
                        'n_synthetic_samples': syn_subset.shape[0],
                        'n_real_samples': real_subset.shape[0],
                        'attribute_distribution_percentage_error': ((syn_subset.shape[0] - real_subset.shape[0]) / real_subset.shape[0]) * 100,
                    }
        
        global_metrics = {
            'mean_dtw': np.nanmean([metrics['mean_dtw'] for metrics in local_metrics.values()]).item(), # ESH: Note np.nanmean instead of np.mean, to handle NaN
            'hellinger_distance': self._hellinger_distance(),
            'attribute_distribution_percentage_error':  np.mean([np.abs(metrics['attribute_distribution_percentage_error']) for metrics in local_metrics.values()]).item()
        }

        self.local_metrics = local_metrics
        self.global_metrics = global_metrics
        
        return global_metrics
    
    def _hellinger_distance(self):
        ''' 
        Compute Hellinger distance to assess attribute distributions.
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

        def normalized_value_counts(array, all_values):
            ''' 
            Utility function to normalize value counts.
            '''
            total_count = len(array)
            value_counts = Counter(array)
            normalized_counts = np.array([value_counts[val] / total_count for val in all_values])
            return normalized_counts

        real_flat = self.real_attribute_vector.flatten()
        syn_flat = self.syn_attribute_vector.flatten()

        all_values = sorted(set(real_flat) | set(syn_flat))

        real_counts = normalized_value_counts(real_flat, all_values)
        syn_counts = normalized_value_counts(syn_flat, all_values)

        distance = hellinger_distance(real_counts, syn_counts)

        return distance
    
    def _combine_vectors(self,
        feature_vector: np.ndarray, attribute_vector: np.ndarray
        ) -> np.ndarray:
        ''' 
        Combine feature and attribute vectors. 

        Args: 
            feature_vector: np.ndarray with features. 
            attribute_vector: np.ndarray with attributes. 

        Returns: 
            combined_data: np.ndarray with attributes and features. 
        '''

        run_length = 656 # config.max_sequence_len 
        
        reshaped_features = feature_vector.reshape(feature_vector.shape[0] * run_length, feature_vector.shape[2])  
        repeated_attributes = np.repeat(attribute_vector, run_length, axis=0)

        stacked_data = np.hstack((reshaped_features, repeated_attributes))

        combined_data = stacked_data.reshape(feature_vector.shape[0], run_length, stacked_data.shape[1])

        return combined_data

    def _define_attributes(self,
        real_attributes: np.ndarray, syn_attributes: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, np.ndarray]:
        ''' 
        Extract unique variables present in attributes. 
        Asserts whether attributes match in real and synthetic data.

        Args: 
            real_attributes: np.ndarray with attributes in real data (str)
            syn_attributes: Optional. np.ndarray with attributes in synthetic data (str). 

        Returns: 
            subjects: np.ndarray with unique subjects. 
            annotations: np.ndarray with unique annotations. 
        '''

        subjects = np.unique(real_attributes[:, -1])
        annotations = np.unique(real_attributes[:, -2])

        if syn_attributes is not None:
            
            syn_subjects = np.unique(syn_attributes[:, -1])
            syn_annotations = np.unique(syn_attributes[:, -2])

            assert np.array_equal(np.sort(subjects), np.sort(syn_subjects)), "Arrays do not contain the same subjects"
            assert np.array_equal(np.sort(annotations), np.sort(syn_annotations)), "Arrays do not contain the same annotations"

        return subjects, annotations


    def _create_sample_subset(self,
        data: np.ndarray, subject: Optional[str] = None, annotation: Optional[str] = None
        ) -> np.ndarray:
        ''' 
        Extract samples for specific attributes.

        Args: 
            data: np.ndarray containing features and attributes. 
            subject: Optional (str). A specific subject from subjects attribute. 
            annotation: Optional (str). A specific annotation from annotations attribute. 

        Returns: 
            sample_subset: np.ndarray of samples corresponding to attribute set,
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
