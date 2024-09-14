
"""
Modified from https://github.com/gretelai/gretel-synthetics/tree/master/src/gretel_synthetics/timeseries_dgan

All changes are indicated with initials: ESH

PyTorch implementation of DoppelGANger, from https://arxiv.org/abs/1909.13403

Based on tensorflow 1 code in https://github.com/fjxmlzn/DoppelGANger

DoppelGANger is a generative adversarial network (GAN) model for time series. It
supports multi-variate time series (referred to as features) and fixed variables
for each time series (attributes). The combination of attribute values and
sequence of feature values is 1 example. Once trained, the model can generate
novel examples that exhibit the same temporal correlations as seen in the
training data. See https://arxiv.org/abs/1909.13403 for additional details on
the model.

"""
import abc
import logging
import math
from collections import Counter
from itertools import cycle
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from errors import DataError, InternalError, ParameterError
from config import DfStyle, DGANConfig, OutputType
from structures import ProgressInfo
from torch_modules import Discriminator, Generator
from transformations import (
    create_additional_attribute_outputs,
    create_outputs_from_data,
    inverse_transform_attributes,
    inverse_transform_features,
    Output,
    transform_attributes,
    transform_features,
)

# ESH
from evaluate import EvaluateTraining
import os 

logger = logging.getLogger(__name__)

AttributeFeaturePair = Tuple[Optional[np.ndarray], list[np.ndarray]]
NumpyArrayTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]

NAN_ERROR_MESSAGE = """
DGAN does not support NaNs, please remove NaNs before training.
"""  

class DGAN:
    """
    DoppelGANger model.

    Interface for training model and generating data based on configuration in
    an DGANConfig instance.

    """

    def __init__(
        self,
        config: DGANConfig,
        attribute_outputs: Optional[List[Output]] = None,
        feature_outputs: Optional[List[Output]] = None,
        # ESH
        fixed_attribute_noise: Optional[torch.Tensor] = None, 
        fixed_feature_noise: Optional[torch.Tensor] = None, 
        attributes: Optional[np.ndarray] = None, 
        features: Optional[np.ndarray] = None,
    ):
        """Create a DoppelGANger model.

        Args:
            config: DGANConfig containing model parameters
            attribute_outputs: custom metadata for attributes, not needed for
                standard usage
            feature_outputs: custom metadata for features, not needed for
                standard usage
        """
        self.config = config

        self.is_built = False

        if config.max_sequence_len % config.sample_len != 0:
            raise ParameterError(
                f"max_sequence_len={config.max_sequence_len} must be divisible by sample_len={config.sample_len}"
            )

        if feature_outputs is not None and attribute_outputs is not None:
            self._build(attribute_outputs, feature_outputs)
        elif feature_outputs is not None or attribute_outputs is not None:
            raise InternalError(
                "feature_outputs and attribute_ouputs must either both be given or both be None"
            )

        self.data_frame_converter = None

        # ESH
        self.fixed_attribute_noise = fixed_attribute_noise
        self.fixed_feature_noise = fixed_feature_noise
        self.attributes = attributes
        self.features = features
        self.epochs_checkpoint = None
    
    ### BUILD MODEL

    def _build(
        self,
        attribute_outputs: Optional[List[Output]],
        feature_outputs: List[Output],
    ):
        """Setup internal structure for DGAN model.

        Args:
            attribute_outputs: custom metadata for attributes
            feature_outputs: custom metadata for features
        """

        self.EPS = 1e-8
        self.attribute_outputs = attribute_outputs
        self.additional_attribute_outputs = create_additional_attribute_outputs(
            feature_outputs
        )
        self.feature_outputs = feature_outputs

        if self.config.cuda and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.generator = Generator(
            attribute_outputs,
            self.additional_attribute_outputs,
            feature_outputs,
            self.config.max_sequence_len,
            self.config.sample_len,
            self.config.attribute_noise_dim,
            self.config.feature_noise_dim,
            self.config.attribute_num_units,
            self.config.attribute_num_layers,
            self.config.feature_num_units,
            self.config.feature_num_layers,
        )

        self.generator.to(self.device, non_blocking=True)

        if self.attribute_outputs is None:
            self.attribute_outputs = []
        attribute_dim = sum(output.dim for output in self.attribute_outputs)

        if not self.additional_attribute_outputs:
            self.additional_attribute_outputs = []
        additional_attribute_dim = sum(
            output.dim for output in self.additional_attribute_outputs
        )
        feature_dim = sum(output.dim for output in feature_outputs)
        self.feature_discriminator = Discriminator(
            attribute_dim
            + additional_attribute_dim
            + self.config.max_sequence_len * feature_dim,
            num_layers=5,
            num_units=200,
        )
        self.feature_discriminator.to(self.device, non_blocking=True)

        self.attribute_discriminator = None
        if not self.additional_attribute_outputs and not self.attribute_outputs:
            self.config.use_attribute_discriminator = False

        if self.config.use_attribute_discriminator:
            self.attribute_discriminator = Discriminator(
                attribute_dim + additional_attribute_dim,
                num_layers=5,
                num_units=200,
            )
            self.attribute_discriminator.to(self.device, non_blocking=True)

        self.attribute_noise_func = lambda batch_size: torch.randn(
            batch_size, self.config.attribute_noise_dim, device=self.device
        )

        self.feature_noise_func = lambda batch_size: torch.randn(
            batch_size,
            self.config.max_sequence_len // self.config.sample_len,
            self.config.feature_noise_dim,
            device=self.device,
        )

        if self.config.forget_bias:

            def init_weights(m):
                if "LSTM" in str(m.__class__):
                    for name, param in m.named_parameters(recurse=False):
                        if "bias_hh" in name:
                            with torch.no_grad():
                                hidden_size = m.hidden_size
                                a = -np.sqrt(1.0 / hidden_size)
                                b = np.sqrt(1.0 / hidden_size)
                                bias_ii = torch.Tensor(hidden_size)
                                bias_ig_io = torch.Tensor(hidden_size * 2)
                                bias_if = torch.Tensor(hidden_size)
                                torch.nn.init.uniform_(bias_ii, a, b)
                                torch.nn.init.uniform_(bias_ig_io, a, b)
                                torch.nn.init.ones_(bias_if)
                                new_param = torch.cat(
                                    [bias_ii, bias_if, bias_ig_io], dim=0
                                )
                                param.copy_(new_param)

            self.generator.apply(init_weights)

        self.is_built = True

    ### TRAIN MODEL 

    def train_numpy(
        self,
        feature_types: Optional[List[OutputType]] = None,
        attribute_types: Optional[List[OutputType]] = None,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        optimization: bool = False # ESH 
    )  -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]: # ESH 
        """Train DGAN model on data in numpy arrays.

        Args:
            feature_types (Optional): Specification of Discrete or Continuous
                type for each variable of the features. If None, assume
                continuous variables for floats and integers, and discrete for
                strings. Ignored if the model was already built, either by
                passing *output params at initialization or because train_* was
                called previously.
            attribute_types (Optional): Specification of Discrete or Continuous
                type for each variable of the attributes. If None, assume
                continuous variables for floats and integers, and discrete for
                strings. Ignored if the model was already built, either by
                passing *output params at initialization or because train_* was
                called previously.
            optimization (ESH): bool. If True, training is conducted with the
                purpose of optimization (grid search).
        
        Returns (ESH):
            losses: losses associated with the generator, discriminator and 
                attribute_discriminator if True.
            metrics: metrics computed using EvaluateTraining (MV-DTW distance, 
                Hellinger distance)
        """
        
        # ESH
        print("Training started...")

        # ESH
        attributes = self.attributes
        features = self.features
        
        if isinstance(features, np.ndarray):
            features = [seq for seq in features]

        logging.info(
            f"features length={len(features)}, first sequence shape={features[0].shape}, dtype={features[0].dtype}",
            extra={"user_log": True},
        )
        if attributes is not None:
            logging.info(
                f"attributes shape={attributes.shape}, dtype={attributes.dtype}",
                extra={"user_log": True},
            )

        if attributes is not None:
            if attributes.shape[0] != len(features):
                raise InternalError(
                    "First dimension of attributes and features must be the same length, i.e., the number of training examples." 
                )

        if attributes is not None and attribute_types is None:
            attribute_types = []
            for i in range(attributes.shape[1]):
                try:
                    attributes[:, i].astype("float")
                    attribute_types.append(OutputType.CONTINUOUS)
                except ValueError:
                    attribute_types.append(OutputType.DISCRETE)

        if feature_types is None:
            feature_types = []
            for i in range(features[0].shape[1]):
                try:
                    for seq in features:
                        seq[:, i].astype("float")
                    feature_types.append(OutputType.CONTINUOUS)
                except ValueError:
                    feature_types.append(OutputType.DISCRETE)

        if not self.is_built:
            logger.info(
                "Determining outputs metadata from input data", extra={"user_log": True}
            )
            attribute_outputs, feature_outputs = create_outputs_from_data(
                attributes,
                features,
                attribute_types,
                feature_types,
                normalization=self.config.normalization,
                apply_feature_scaling=self.config.apply_feature_scaling,
                apply_example_scaling=self.config.apply_example_scaling,
            )
            logger.info("Building DGAN networks", extra={"user_log": True})
            self._build(
                attribute_outputs,
                feature_outputs,
            )

        continuous_features_ind = [
            ind
            for ind, val in enumerate(self.feature_outputs)
            if "ContinuousOutput" in str(val.__class__)
        ]

        if continuous_features_ind:

            logger.info(
                f"Checking for nans in the {len(continuous_features_ind)} numeric columns",
                extra={"user_log": True},
            )

            valid_examples = validation_check(
                features,
                continuous_features_ind,
            )

            features = [seq for valid, seq in zip(valid_examples, features) if valid]
            if attributes is not None:
                attributes = attributes[valid_examples]

            logger.info(
                "Applying linear interpolations for nans (does not mean nans are present)",
                extra={"user_log": True},
            )

            nan_linear_interpolation(features, continuous_features_ind)

        logger.info("Creating encoded array of features", extra={"user_log": True})
        (
            internal_features,
            internal_additional_attributes,
        ) = transform_features(
            features, self.feature_outputs, self.config.max_sequence_len
        )

        if internal_additional_attributes is not None:
            if np.any(np.isnan(internal_additional_attributes)):
                raise InternalError(
                    f"NaN found in internal additional attributes. {NAN_ERROR_MESSAGE}"
                )
        else:
            internal_additional_attributes = np.full(
                (internal_features.shape[0], 1), np.nan, dtype=np.float32
            )

        logger.info("Creating encoded array of attributes", extra={"user_log": True})
        if attributes is not None and self.attribute_outputs is not None:
            internal_attributes = transform_attributes(
                attributes,
                self.attribute_outputs,
            )
        else:
            internal_attributes = np.full(
                (internal_features.shape[0], 1), np.nan, dtype=np.float32
            )

        logger.info(
            f"internal_features shape={internal_features.shape}, dtype={internal_features.dtype}",
            extra={"user_log": True},
        )
        logger.info(
            f"internal_additional_attributes shape={internal_additional_attributes.shape}, dtype={internal_additional_attributes.dtype}",
            extra={"user_log": True},
        )
        logger.info(
            f"internal_attributes shape={internal_attributes.shape}, dtype={internal_attributes.dtype}",
            extra={"user_log": True},
        )

        if self.attribute_outputs and np.any(np.isnan(internal_attributes)):
            raise InternalError(
                f"NaN found in internal attributes. {NAN_ERROR_MESSAGE}"
            )

        logger.info("Creating TensorDataset", extra={"user_log": True})
        dataset = TensorDataset(
            torch.Tensor(internal_attributes),
            torch.Tensor(internal_additional_attributes),
            torch.Tensor(internal_features),
        )

        logger.info("Calling _train()", extra={"user_log": True})
        losses, metrics = self._train(
            dataset, 
            progress_callback=progress_callback, 
            optimization=optimization) # ESH 

        return losses, metrics # ESH 
    
    def _train(
        self,
        dataset: Dataset,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        optimization: bool = False # ESH
    ) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]: # ESH
        """
        Internal method for training DGAN model.

        ESH: Added functionality here includes:
            - Saving model checkpoint every epoch
            - Saving loss for all networks every epoch
            - Saving synthetic data every 10 epochs after epoch 500
            - Early stopping based on mean_dtw metric, if improvements are not
                seen for 100 consecutive epochs after epoch 500
            - Continuing training from last epoch if training has been interrupted

        Args:
            dataset: torch Dataset containing tuple of (attributes, 
                additional_attributes, features)
            optimization (ESH): bool. If True, training is conducted with the
                purpose of optimization (grid search).

        Returns (ESH): 
            losses: losses associated with the generator, discriminator and 
                attribute_discriminator if True.
            metrics: metrics computed using EvaluateTraining (MV-DTW distance, 
                Hellinger distance)
        """
        if len(dataset) <= 1:
            raise DataError(
                f"DGAN requires multiple examples to train, received {len(dataset)} example."
                + "Consider splitting a single long sequence into many subsequences to obtain "
                + "multiple examples for training."
            )

        drop_last = len(dataset) % self.config.batch_size == 1

        loader = DataLoader(
            dataset,
            self.config.batch_size,
            shuffle=True,
            drop_last=drop_last,
            pin_memory=True,
        )

        opt_discriminator = torch.optim.RAdam( # ESH
            self.feature_discriminator.parameters(),
            lr=self.config.discriminator_learning_rate,
            betas=(self.config.discriminator_beta1, 0.999),
        )

        opt_attribute_discriminator = None
        if self.attribute_discriminator is not None:
            opt_attribute_discriminator = torch.optim.RAdam( # ESH
                self.attribute_discriminator.parameters(),
                lr=self.config.attribute_discriminator_learning_rate,
                betas=(self.config.attribute_discriminator_beta1, 0.999),
            )

        opt_generator = torch.optim.RAdam( # ESH
            self.generator.parameters(),
            lr=self.config.generator_learning_rate,
            betas=(self.config.generator_beta1, 0.999),
        )

        global_step = 0

        self._set_mode(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision_training)

        # ESH
        metrics = {'epoch': [], 'mean_dtw': [], 'hellinger_distance': []} #'average_attribute_error': [], removed from middle - used in optimization
        losses = {'epoch': [], 'discriminator': [], 'generator': [], 'attribute_discriminator': []}

        # ESH
        if optimization:
            early_stop_counter_optim = 0
            should_stop = False
        else:
            early_stop_counter = 0
            best_mean_dtw = float('inf') 
            patience = 10 
        
        # ESH - should be configurable
        checkpoint_dir = "/content/drive/MyDrive/Speciale/results/model_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # ESH
        start_epoch = self.epochs_checkpoint + 1 if self.epochs_checkpoint is not None else 0
    
        for epoch in range(start_epoch, self.config.epochs): #for epoch in range(self.config.epochs):
            logger.info(f"epoch: {epoch}")

            # ESH
            self.epochs_checkpoint += 1

            # ESH
            epoch_discriminator_losses = []
            epoch_generator_losses = []
            epoch_attribute_discriminator_losses = []

            for batch_idx, real_batch in enumerate(loader):
                global_step += 1

                with torch.cuda.amp.autocast(
                    enabled=self.config.mixed_precision_training
                ):
                    attribute_noise = self.attribute_noise_func(real_batch[0].shape[0])
                    feature_noise = self.feature_noise_func(real_batch[0].shape[0])

                    generated_batch = self.generator(attribute_noise, feature_noise)
                    real_batch = [
                        x.to(self.device, non_blocking=True) for x in real_batch
                    ]

                for _ in range(self.config.discriminator_rounds):
                    opt_discriminator.zero_grad(
                        set_to_none=self.config.mixed_precision_training
                    )
                    with torch.cuda.amp.autocast(enabled=True):
                        generated_output = self._discriminate(generated_batch)
                        real_output = self._discriminate(real_batch)

                        loss_generated = torch.mean(generated_output)
                        loss_real = -torch.mean(real_output)
                        loss_gradient_penalty = self._get_gradient_penalty(
                            generated_batch, real_batch, self._discriminate
                        )

                        loss = (
                            loss_generated
                            + loss_real
                            + self.config.gradient_penalty_coef * loss_gradient_penalty
                        )

                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(opt_discriminator)
                    scaler.update()

                    # ESH
                    epoch_discriminator_losses.append(loss.item())

                    if opt_attribute_discriminator is not None:
                        opt_attribute_discriminator.zero_grad(set_to_none=False)

                        with torch.cuda.amp.autocast(
                            enabled=self.config.mixed_precision_training
                        ):
                            generated_output = self._discriminate_attributes(
                                generated_batch[:-1]
                            )
                            real_output = self._discriminate_attributes(real_batch[:-1])

                            loss_generated = torch.mean(generated_output)
                            loss_real = -torch.mean(real_output)
                            loss_gradient_penalty = self._get_gradient_penalty(
                                generated_batch[:-1],
                                real_batch[:-1],
                                self._discriminate_attributes,
                            )

                            attribute_loss = (
                                loss_generated
                                + loss_real
                                + self.config.attribute_gradient_penalty_coef
                                * loss_gradient_penalty
                            )

                        scaler.scale(attribute_loss).backward(retain_graph=True)
                        scaler.step(opt_attribute_discriminator)
                        scaler.update()

                        # ESH
                        epoch_attribute_discriminator_losses.append(attribute_loss.item())

                for _ in range(self.config.generator_rounds):
                    opt_generator.zero_grad(set_to_none=False)
                    with torch.cuda.amp.autocast(
                        enabled=self.config.mixed_precision_training
                    ):
                        generated_output = self._discriminate(generated_batch)

                        if self.attribute_discriminator:

                            attribute_generated_output = self._discriminate_attributes(
                                generated_batch[:-1]
                            )

                            loss = -torch.mean(
                                generated_output
                            ) + self.config.attribute_loss_coef * -torch.mean(
                                attribute_generated_output
                            )
                        else:
                            loss = -torch.mean(generated_output)

                    scaler.scale(loss).backward()
                    scaler.step(opt_generator)
                    scaler.update()

                    # ESH
                    epoch_generator_losses.append(loss.item())

                if progress_callback is not None:
                    progress_callback(
                        ProgressInfo(
                            epoch=epoch,
                            total_epochs=self.config.epochs,
                            batch=batch_idx,
                            total_batches=len(loader),
                        )
                    )

            # ESH
            losses['epoch'].append(epoch)
            losses['discriminator'].append(np.mean(epoch_discriminator_losses))
            losses['generator'].append(np.mean(epoch_generator_losses))
            if epoch_attribute_discriminator_losses:
                losses['attribute_discriminator'].append(np.mean(epoch_attribute_discriminator_losses))
            else:
                losses['attribute_discriminator'].append(0)

            # ESH
            checkpoint_model_path = os.path.join(checkpoint_dir, "checkpoint_model.pth")
            self.save(checkpoint_model_path)
            
            # ESH 
            if optimization:
                self._set_mode(False)
                generated_attributes, generated_features = self.generate_numpy(
                    attribute_noise=self.fixed_attribute_noise, 
                    feature_noise=self.fixed_feature_noise)
                eval_object = EvaluateTraining(
                    real_feature_vector=self.features, 
                    real_attribute_vector=self.attributes, 
                    syn_feature_vector=generated_features, 
                    syn_attribute_vector=generated_attributes)
                global_metrics = eval_object.evaluate()
                metrics['epoch'].append(epoch)
                metrics['mean_dtw'].append(global_metrics['mean_dtw'])
                metrics['average_attribute_error'].append(global_metrics['attribute_distribution_percentage_error'])
                
                if metrics['mean_dtw'][0]  > 3000: # Threshold based on experimentation
                    early_stop_counter_optim += 1
                else:
                    early_stop_counter_optim = 0

                if early_stop_counter_optim >= 10:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    should_stop = True
                    break

                self._set_mode(True)

            else:
                print(f"Epoch [{epoch}] completed." if epoch % 10 == 0 else "", end="")
                print()
                if epoch >= 500 and epoch % 10 == 0:
                    self._set_mode(False)
                    checkpoint_attributes, checkpoint_features = self.generate_numpy(
                        attribute_noise=self.fixed_attribute_noise, 
                        feature_noise=self.fixed_feature_noise)
                
                    os.makedirs("/content/drive/My Drive/Speciale/data_synthetic", exist_ok=True)
                    checkpoint_name = f"checkpoint_{epoch}"
                    attribute_path = os.path.join("/content/drive/My Drive/Speciale/data_synthetic", f"synthetic_attributes_{checkpoint_name}.npy")
                    feature_path = os.path.join("/content/drive/My Drive/Speciale/data_synthetic", f"synthetic_features_{checkpoint_name}.npy")
                    np.save(attribute_path, checkpoint_attributes)
                    np.save(feature_path, checkpoint_features)
                    print(f"Checkpoint data saved at epoch {epoch}")

                    eval_object = EvaluateTraining(
                        real_feature_vector=self.features,
                        real_attribute_vector=self.attributes,
                        syn_feature_vector=checkpoint_features,
                        syn_attribute_vector=checkpoint_attributes
                    )

                    global_metrics = eval_object.evaluate()
                    metrics['epoch'].append(epoch)
                    metrics['mean_dtw'].append(global_metrics['mean_dtw'])
                    metrics['hellinger_distance'].append(global_metrics['hellinger_distance'])

                    current_mean_dtw = global_metrics['mean_dtw']
        
                    if current_mean_dtw < best_mean_dtw:
                        best_mean_dtw = current_mean_dtw
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1 
                    
                    if early_stop_counter >= patience:
                        print(f"Stopping early at epoch [{epoch}] as mean_dtw did not improve for {patience} consecutive evaluations.")
                        break

                    self._set_mode(True)
            
        return losses, metrics

    ### GENERATE DATA

    def generate_numpy(
        self,
        n: Optional[int] = None,
        attribute_noise: Optional[torch.Tensor] = None,
        feature_noise: Optional[torch.Tensor] = None,
    ) -> AttributeFeaturePair:
        """Generate synthetic data from DGAN model.

        Once trained, a DGAN model can generate arbitrary amounts of
        synthetic data by sampling from the noise distributions. Specify either
        the number of records to generate, or the specific noise vectors to use.

        Args:
            n: number of examples to generate
            attribute_noise: noise vectors to create synthetic data
            feature_noise: noise vectors to create synthetic data

        Returns:
            Tuple of attributes and features as numpy arrays.
        """

        if not self.is_built:
            raise InternalError("Must build DGAN model prior to generating samples.")

        if n is not None:

            num_batches = n // self.config.batch_size
            if n % self.config.batch_size != 0:
                num_batches += 1

            internal_data_list = []
            for _ in range(num_batches):
                internal_data_list.append(
                    self._generate(
                        self.attribute_noise_func(self.config.batch_size),
                        self.feature_noise_func(self.config.batch_size),
                    )
                )

            internal_data = tuple(
                (
                    np.concatenate(d, axis=0)
                    if not (np.array(d) == None).any() 
                    else None
                )
                for d in zip(*internal_data_list)
            )

        else:
            if attribute_noise is None or feature_noise is None:
                raise InternalError(
                    "generate() must receive either n or both attribute_noise and feature_noise"
                )
            attribute_noise = attribute_noise.to(self.device, non_blocking=True)
            feature_noise = feature_noise.to(self.device, non_blocking=True)

            internal_data = self._generate(attribute_noise, feature_noise)

        (
            internal_attributes,
            internal_additional_attributes,
            internal_features,
        ) = internal_data

        attributes = None
        if internal_attributes is not None and self.attribute_outputs is not None:
            attributes = inverse_transform_attributes(
                internal_attributes,
                self.attribute_outputs,
            )

        if internal_features is None:
            raise InternalError(
                "Received None instead of internal features numpy array"
            )

        features = inverse_transform_features(
            internal_features,
            self.feature_outputs,
            additional_attributes=internal_additional_attributes,
        )

        features = [seq for seq in features]

        if n is not None:
            if attributes is None:
                features = features[:n]
                return None, features
            else:
                return attributes[:n], features[:n]
        
        # ESH
        features = np.array(features)

        return attributes, features

    def _generate(
        self, attribute_noise: torch.Tensor, feature_noise: torch.Tensor
    ) -> NumpyArrayTriple:
        """Internal method for generating from a DGAN model.

        Returns data in the internal representation, including additional
        attributes for the midpoint and half-range for features when
        apply_example_scaling is True for some features.

        Args:
            attribute_noise: noise vectors to create synthetic data
            feature_noise: noise vectors to create synthetic data

        Returns:
            Tuple of generated data in internal representation. If additional
            attributes are used in the model, the tuple is 3 elements:
            attributes, additional_attributes, features. If there are no
            additional attributes in the model, the tuple is 2 elements:
            attributes, features.
        """
        self._set_mode(False)
        batch = self.generator(attribute_noise, feature_noise)
        return tuple(t.cpu().detach().numpy() for t in batch)

    ### INTERNAL METHODS

    def _discriminate(
        self,
        batch,
    ) -> torch.Tensor:
        """Internal helper function to apply the GAN discriminator.

        Args:
            batch: internal data representation

        Returns:
            Output of the GAN discriminator.
        """

        batch = [index for index in batch if not torch.isnan(index).any()]
        inputs = list(batch)

        inputs[-1] = torch.reshape(inputs[-1], (inputs[-1].shape[0], -1))

        input = torch.cat(inputs, dim=1)

        output = self.feature_discriminator(input)
        return output

    def _discriminate_attributes(self, batch) -> torch.Tensor:
        """Internal helper function to apply the GAN attribute discriminator.

        Args:
            batch: tuple of internal data of size 2 elements
            containing attributes and additional_attributes.

        Returns:
            Output for GAN attribute discriminator.
        """
        batch = [index for index in batch if not torch.isnan(index).any()]
        if not self.attribute_discriminator:
            raise InternalError(
                "discriminate_attributes called with no attribute_discriminator"
            )

        input = torch.cat(batch, dim=1)

        output = self.attribute_discriminator(input)
        return output

    def _get_gradient_penalty(
        self, generated_batch, real_batch, discriminator_func
    ) -> torch.Tensor:
        """Internal helper function to compute the gradient penalty component of
        DoppelGANger loss.

        Args:
            generated_batch: internal data from the generator
            real_batch: internal data for the training batch
            discriminator_func: function to apply discriminator to interpolated
                data

        Returns:
            Gradient penalty tensor.
        """
        generated_batch = [
            generated_index
            for generated_index in generated_batch
            if not torch.isnan(generated_index).any()
        ]
        real_batch = [
            real_index for real_index in real_batch if not torch.isnan(real_index).any()
        ]

        alpha = torch.rand(generated_batch[0].shape[0], device=self.device)
        interpolated_batch = [
            self._interpolate(g, r, alpha).requires_grad_(True)
            for g, r in zip(generated_batch, real_batch)
        ]

        interpolated_output = discriminator_func(interpolated_batch)

        gradients = torch.autograd.grad(
            interpolated_output,
            interpolated_batch,
            grad_outputs=torch.ones(interpolated_output.shape, device=self.device),
            retain_graph=True,
            create_graph=True,
        )

        squared_sums = [
            torch.sum(torch.square(g.view(g.size(0), -1))) for g in gradients
        ]

        norm = torch.sqrt(sum(squared_sums) + self.EPS)

        return ((norm - 1.0) ** 2).mean()

    def _interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Internal helper function to interpolate between 2 tensors.

        Args:
            x1: tensor
            x2: tensor
            alpha: scale or 1d tensor with values in [0,1]

        Returns:
            x1 + alpha * (x2 - x1)
        """
        diff = x2 - x1
        expanded_dims = [1 for _ in diff.shape]
        expanded_dims[0] = -1
        reshaped_alpha = alpha.reshape(expanded_dims).expand(diff.shape)

        return x1 + reshaped_alpha * diff

    def _set_mode(self, mode: bool = True):
        """Set torch module training mode.

        Args:
            train_mode: whether to set training mode (True) or evaluation mode
                (False). Default: True
        """
        self.generator.train(mode)
        self.feature_discriminator.train(mode)
        if self.attribute_discriminator:
            self.attribute_discriminator.train(mode)

    ### SAVE AND LOAD MODEL FUNCTIONS

    def save(self, file_name: str, **kwargs):
        """Save DGAN model to a file.

        Args:
            file_name: location to save serialized model
            kwargs: additional parameters passed to torch.save
        """
        state = {
            "config": self.config.to_dict(),
            "attribute_outputs": self.attribute_outputs,
            "feature_outputs": self.feature_outputs,
        }
        state["generate_state_dict"] = self.generator.state_dict()
        state["feature_discriminator_state_dict"] = (
            self.feature_discriminator.state_dict()
        )
        if self.attribute_discriminator is not None:
            state["attribute_discriminator_state_dict"] = (
                self.attribute_discriminator.state_dict()
            )

        if self.data_frame_converter is not None:
            state["data_frame_converter"] = self.data_frame_converter.state_dict()

        torch.save(state, file_name, **kwargs)

    @classmethod
    def load(cls, file_name: str, **kwargs) -> DGAN:
        """Load DGAN model instance from a file.

        Args:
            file_name: location to load from
            kwargs: additional parameters passed to torch.load, for example, use
                map_location=torch.device("cpu") to load a model saved for GPU on
                a machine without cuda

        Returns:
            DGAN model instance
        """

        state = torch.load(file_name, **kwargs)

        config = DGANConfig(**state["config"])
        dgan = DGAN(config)

        dgan._build(state["attribute_outputs"], state["feature_outputs"])

        dgan.generator.load_state_dict(state["generate_state_dict"])
        dgan.feature_discriminator.load_state_dict(
            state["feature_discriminator_state_dict"]
        )
        if "attribute_discriminator_state_dict" in state:
            if dgan.attribute_discriminator is None:
                raise InternalError(
                    "Error deserializing model: found unexpected attribute discriminator state in file"
                )

            dgan.attribute_discriminator.load_state_dict(
                state["attribute_discriminator_state_dict"]
            )

        return dgan
    

def find_max_consecutive_nans(array: np.ndarray) -> int:
    """
    Returns the maximum number of consecutive NaNs in an array.

    Args:
        array: 1-d numpy array of time series per example.

    Returns:
        max_cons_nan: The maximum number of consecutive NaNs in a times series array.

    """
    max_cons_nan = np.max(
        np.diff(np.concatenate(([-1], np.where(~np.isnan(array))[0], [len(array)]))) - 1
    )
    return max_cons_nan


def validation_check(
    features: list[np.ndarray],
    continuous_features_ind: list[int],
    invalid_examples_ratio_cutoff: float = 0.5,
    nans_ratio_cutoff: float = 0.1,
    consecutive_nans_max: int = 5,
    consecutive_nans_ratio_cutoff: float = 0.05,
) -> np.ndarray:
    """Checks if continuous features of examples are valid.

    Returns a 1-d numpy array of booleans with shape (#examples) indicating
    valid examples.
    Examples with continuous features fall into 3 categories: good, valid (fixable) and
    invalid (non-fixable).
    - "Good" examples have no NaNs.
    - "Valid" examples have a low percentage of nans and a below a threshold number of
    consecutive NaNs.
    - "Invalid" are the rest, and are marked "False" in the returned array.  Later on,
    these are omitted from training. If there are too many, later, we error out.

    Args:
        features: list of 2-d numpy arrays, each element is a sequence of
            possibly varying length
        continuous_features_ind: list of indices of continuous features to
            analyze, indexes the 2nd dimension of the sequence arrays in
            features
        invalid_examples_ratio_cutoff: Error out if the invalid examples ratio
            in the dataset is higher than this value.
        nans_ratio_cutoff: If the percentage of nans for any continuous feature
           in an example is greater than this value, the example is invalid.
        consecutive_nans_max: If the maximum number of consecutive nans in a
           continuous feature is greater than this number, then that example is
           invalid.
        consecutive_nans_ratio_cutoff: If the maximum number of consecutive nans
            in a continuous feature is greater than this ratio times the length of
            the example (number samples), then the example is invalid.

    Returns:
        valid_examples: 1-d numpy array of booleans indicating valid examples with
        shape (#examples).

    """

    nan_ratio_feature = np.array(
        [
            [
                np.mean(np.isnan(seq[:, ind].astype("float")))
                for ind in continuous_features_ind
            ]
            for seq in features
        ]
    )

    nan_ratio = nan_ratio_feature < nans_ratio_cutoff

    cons_nans_feature = np.array(
        [
            [
                find_max_consecutive_nans(seq[:, ind].astype("float"))
                for ind in continuous_features_ind
            ]
            for seq in features
        ]
    )

    cons_nans_threshold = np.clip(
        [consecutive_nans_ratio_cutoff * seq.shape[0] for seq in features],
        a_min=2,
        a_max=consecutive_nans_max,
    ).reshape((-1, 1))
    cons_nans = cons_nans_feature < cons_nans_threshold

    valid_examples_per_feature = np.logical_and(nan_ratio, cons_nans)
    valid_examples = np.all(valid_examples_per_feature, axis=1)

    if np.mean(valid_examples) < invalid_examples_ratio_cutoff:
        raise DataError(
            f"More than {100*invalid_examples_ratio_cutoff}% invalid examples in the continuous features. Please reduce the ratio of the NaNs and try again!" 
        )

    if (~valid_examples).any():
        logger.warning(
            f"There are {sum(~valid_examples)} examples that have too many nan values in numeric features, accounting for {np.mean(~valid_examples)*100}% of all examples. These invalid examples will be omitted from training.", 
            extra={"user_log": True},
        )

    return valid_examples


def nan_linear_interpolation(
    features: list[np.ndarray], continuous_features_ind: list[int]
):
    """Replaces all NaNs via linear interpolation.

    Changes numpy arrays in features in place.

    Args:
        features: list of 2-d numpy arrays, each element is a sequence of shape
            (sequence_len, #features)
        continuous_features_ind: features to apply nan interpolation to, indexes
            the 2nd dimension of the sequence arrays of features
    """
    for seq in features:
        for ind in continuous_features_ind:
            continuous_feature = seq[:, ind].astype("float")
            is_nan = np.isnan(continuous_feature)
            if is_nan.any():
                ind_func = lambda z: z.nonzero()[0]
                seq[is_nan, ind] = np.interp(
                    ind_func(is_nan), ind_func(~is_nan), continuous_feature[~is_nan]
                )
