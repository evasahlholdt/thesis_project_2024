from typing import Dict, Optional, Tuple, Any
import time
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, AveragePooling2D, Dropout, Flatten, Dense, concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks, backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

class ShallowConvNet:
    def __init__(self, 
    dataset_name: str,
    nr_channels: int, 
    sequence_length: int,
    use_cpu: bool = False
    ):
        '''
        Class and methods relating to classification model ShallowConvNet. 

        The implementation is credited to the Army Research Laboratory (ARL)
        Source: https://github.com/vlawhern/arl-eegmodels

        Code is modified heavily for the current implementation; however, model structure and parameters
        are identical to original implementation by Roots et al. (2020).

        Source: https://github.com/rootskar/EEGMotorImagery 

        Args: 
            dataset_name: str. Name for dataset used for classification.
            nr_channels: int. Number of EEG channels in dataset used.
            sequence_length: int. Length of EEG series.
            use_cpu: bool. True if using use_cpu, False if GPU. 
        '''
        self.nr_classes = 2
        self.dataset_name = dataset_name
        self.nr_channels = nr_channels
        self.sequence_length = sequence_length
        self.dropout_rate = 0 # ESH: Changed from 0.5 to avoid regularization
        self.use_cpu = use_cpu
        self.model = self.build_model()

    def build_model(self
    ) -> Model:
        '''
        Build and compile the ShallowConvNet model.

        This method constructs the ShallowConvNet model architecture based on whether 
        the model is to be trained using a CPU or GPU. It sets the input shape, 
        convolutional filters, pooling size, and other parameters accordingly.

        Returns:
            Model: A compiled Keras model ready for training.
        '''
        if self.use_cpu:
            input_shape = (self.sequence_length, self.nr_channels, 1)
            conv_filters = (25, 1)
            conv_filters2 = (1, self.nr_channels)
            pool_size = (45, 1)
            strides = (15, 1)
            axis = -1
        else:
            input_shape = (1, self.nr_channels, self.sequence_length)
            conv_filters = (1, 20)
            conv_filters2 = (self.nr_channels, 1)
            pool_size = (1, 45)
            strides = (1, 15)
            axis = 1

        input_main = Input(input_shape)
        block1 = Conv2D(20, conv_filters,
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(20, conv_filters2, use_bias=False,
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(axis=axis, epsilon=1e-05, momentum=0.1)(block1)
        block1 = Activation(self.square)(block1)
        block1 = AveragePooling2D(pool_size=pool_size, strides=strides)(block1)
        block1 = Activation(self.log)(block1)
        block1 = Dropout(self.dropout_rate)(block1)
        flatten = Flatten()(block1)
        dense = Dense(self.nr_classes, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        return Model(inputs=input_main, outputs=softmax)

    @staticmethod
    def square(x):
        return tf.math.square(x)

    @staticmethod
    def log(x):
        return tf.math.log(tf.clip_by_value(x, 1e-10, x))

    def get_model(self):
        return self.model

    def _predict_accuracy(self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        '''
        Predict the accuracy of the model on the test data.

        Args:
            X_test (np.ndarray): The input features for the test dataset.
            y_test (np.ndarray): The true labels for the test dataset.

        Returns:
            Tuple[float, np.ndarray, np.ndarray]: 
                - The accuracy of the model on the test dataset.
                - A boolean array indicating which predictions were correct.
                - The predicted labels for the test dataset.
        '''

        probs = self.model.predict(X_test)

        preds = probs.argmax(axis=-1)
        equals = preds == y_test.argmax(axis=-1)
        acc = np.mean(equals)

        print("Classification accuracy for %s : %f " % (self.dataset_name, acc))

        return acc, equals, preds

    def train_test(self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        nr_epochs: Optional[int] = 50
    ) -> Dict[str, Any]:
        '''
        Train and test the ShallowConvNet model.

        Calculates performance metrics (accuracy, precision, recall, and F1 score), 
        confusion matrix, training/testing times.

        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training attributes.
            X_test (np.ndarray): Testing features.
            y_test (np.ndarray): Testing attributes.
            nr_epochs (int, optional): Number of epochs for training. Default is 50.

        Returns:
            results: Dict[str, Any]. A dictionary containing the results, including accuracy,
                            precision, recall, F1 scores, confusion matrix, and 
                            training/testing times.
        '''
        acc = 0
        equals = []

        X_val, y_val = [], []
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                        test_size = 0.5, random_state=42)

        self.model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=['accuracy']) 

        training_start = time.time()

        history = self.model.fit(X_train, y_train, batch_size=64, shuffle=True, epochs=nr_epochs,
                                validation_data=(X_val, y_val), verbose=False)
        
        training_total_time = time.time() - training_start
        print("Training completed")
        print("Model {} total training time was {} seconds".format(self.dataset_name, training_total_time))
        
        testing_start = time.time()
        acc, equals, preds = self._predict_accuracy(X_test, y_test)
        testing_total_time = time.time() - testing_start
        print("Model {} total testing time was {} seconds".format(self.dataset_name, testing_total_time))
        
        labels = np.argmax(y_test, axis=1)

        conf_mat = confusion_matrix(labels, preds)

        precision_left = precision_score(labels, preds, average='binary', pos_label=0)
        print('Precision for left hand/real samples: %.3f' % precision_left)

        precision_right = precision_score(labels, preds, average='binary', pos_label=1)
        print('Precision for right hand/syn samples: %.3f' % precision_right)

        recall_left = recall_score(labels, preds, average='binary', pos_label=0)
        print('Recall for left hand/real samples: %.3f' % recall_left)

        recall_right = recall_score(labels, preds, average='binary', pos_label=1)
        print('Recall for right hand/syn samples: %.3f' % recall_right)

        f1_left = f1_score(labels, preds, pos_label=0, average='binary')
        print('F1-Score for left hand/real samples: %.3f' % f1_left)
        
        f1_right = f1_score(labels, preds, pos_label=1, average='binary')
        print('F1-Score for right hand/syn samples: %.3f' % f1_right)

        results = {
            'accuracy': acc,
            'precision_left_real': precision_left,
            'precision_right_syn': precision_right,
            'recall_left_real': recall_left,
            'recall_right_syn': recall_right,
            'f1_left_real': f1_left,
            'f1_right_syn': f1_right,
            'confusion_matrix': conf_mat,
            'training_time': training_total_time,
            'testing_time': testing_total_time
        }

        return results
