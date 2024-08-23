#!/usr/bin/env python
# coding: utf-8

from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.simulation import GenerativeModel
from bayesflow.simulation import Prior
from bayesflow.summary_networks import SequenceNetwork, SplitNetwork, TimeSeriesTransformer
from bayesflow.trainers import Trainer


def split_data(i: int, x: tf.Tensor):
    """
    i index of network, x inputs to the SplitNetwork
    # Should return the input for the corresponding network
    """
    selector = tf.where(x[:, :, -1] == i, 1.0, 0.0)
    x0 = x[:, :, 0] * selector
    x1 = x[:, :, 1] * selector
    split_x = tf.stack((selector, x0, x1), axis=-1)
    return split_x


def split_data_2d(i: int, x: tf.Tensor):
    """
    i index of network, x inputs to the SplitNetwork, now with two measurements per observation
    # Should return the input for the corresponding network
    """
    selector = tf.where(x[:, :, -1] == i, 1.0, 0.0)
    x0 = x[:, :, 0] * selector
    x1 = x[:, :, 1] * selector
    x2 = x[:, :, 2] * selector
    split_x = tf.stack((selector, x0, x1, x2), axis=-1)
    return split_x


class NlmeBaseAmortizer(ABC):
    """
       This model class contains all information needed for simulation, individual specific posterior inference.
    """

    def __init__(self, name: str,
                 param_names: list,
                 network_idx: int = -1,
                 load_best: bool = False,
                 prior_mean: Optional[np.ndarray] = None,
                 prior_cov: Optional[np.ndarray] = None,
                 prior_type: str = 'normal',
                 max_n_obs: Optional[int] = None,
                 changeable_obs_n: bool = False):

        self.name = name
        # define names of parameters
        self.param_names = param_names
        self.log_param_names = ['$\log$ ' + name for name in self.param_names]
        self.n_params = len(self.param_names)

        # define prior values (for log-parameters)
        self.prior_mean: np.ndarray = np.empty(self.n_params) if prior_mean is None else prior_mean
        self.prior_cov: np.ndarray = np.empty(self.n_params) if prior_cov is None else prior_cov
        self.prior_std: np.ndarray = np.empty(self.n_params) if prior_cov is None else np.sqrt(np.diag(prior_cov))
        self.prior_type = prior_type

        # define maximal number of observations
        self.max_n_obs = 180 if max_n_obs is None else max_n_obs
        self.changeable_obs_n = changeable_obs_n  # if True, the number of observations may change per simulation
        self.n_obs_per_measure = 1  # number of observations per measurement, only important if inputs can be split

        # training parameters
        self.n_epochs = 500
        self.n_coupling_layers = 6
        self.n_dense_layers_in_coupling = 2
        self.coupling_design = ['affine', 'spline'][0]
        self.batch_size = 128
        self.bidirectional_LSTM = False  # default
        self.summary_dim = 10  # default
        self.num_conv_layers = 2  # default
        self.summary_network_type = ['sequence', 'split-sequence', 'transformer'][0]
        self.latent_dist = 'normal'  # default

        # amortizer and prior
        # must be before calling the generative model so prior etc. are up-to-date
        self.network_name = self.load_amortizer_configuration(model_idx=network_idx, load_best=load_best)
        self.amortizer = None
        self._build_amortizer()
        self._build_prior()
        self.configured_input = partial(configure_input,
                                        prior_means=self.prior_mean,
                                        prior_stds=self.prior_std)

        self.simulator = None  # to be defined in child class
        """
        # to create trainer call
        self.build_trainer()
        """

    @abstractmethod
    def load_amortizer_configuration(self, model_idx: int = -1, load_best: bool = False) -> Optional[str]:
        # load trained model
        if model_idx == 0:
            self.n_epochs = 500
            self.n_coupling_layers = 5
            model_name = 'model_XY'
            raise NotImplementedError('Model not implemented yet.')
        else:
            model_name = None
        return model_name

    def build_trainer(self, path_store_network: str, max_to_keep: int = 7) -> Trainer:
        if self.simulator is None:
            raise ValueError('Simulator not defined yet.')
        generative_model = GenerativeModel(prior=self.prior,
                                           simulator=self.simulator,
                                           name=self.name)

        # build the trainer with networks and generative model
        trainer = Trainer(amortizer=self.amortizer,
                          generative_model=generative_model,
                          configurator=self.configured_input,
                          # When using transformers as summary networks, you should use a smaller learning rate
                          default_lr=0.0005 if self.summary_network_type != 'transformer' else 1e-5,
                          checkpoint_path=path_store_network,
                          max_to_keep=max_to_keep)

        trainer.load_pretrained_network()
        history = trainer.loss_history.get_plottable()

        if 'val_losses' in history:
            # Find the checkpoint with the lowest validation loss out of the last 7
            max_to_keep = min(max_to_keep, len(history['val_losses']))
            recent_losses = history['val_losses'].iloc[-(max_to_keep-1):]
            best_valid_epoch = recent_losses['Loss'].idxmin() + 1  # checkpoints are 1-based indexed
            new_checkpoint = trainer.manager.latest_checkpoint.rsplit('-', 1)[0] + f'-{best_valid_epoch}'
            trainer.checkpoint.restore(new_checkpoint)
            print("Networks loaded from {}".format(new_checkpoint))
            print("Validation loss: {:.4f}".format(recent_losses['Loss'].min()))

        print(self.amortizer.summary())
        return trainer

    def _build_amortizer(self) -> None:
        """
        build amortizer, i.e., neural networks (summary and inference networks)

        Returns: bayesflow.amortizers.AmortizedPosterior -- the amortizer
        """
        # summary network
        # 2^k hidden units s.t. 2^k > #datapoints
        lstm_units = 2 ** int(np.ceil(np.log2(self.max_n_obs)))
        lstm_units = max(lstm_units, 32)

        if self.summary_network_type == 'split-sequence':
            network_kwargs = {'summary_dim': self.summary_dim,
                              'num_conv_layers': self.num_conv_layers,
                              'lstm_units': lstm_units,
                              'bidirectional': self.bidirectional_LSTM}
            print(f'using a split network with 2 splits, '
                  f'in each {self.num_conv_layers} layers of MultiConv1D, '
                  f'a {"bidirectional" if self.bidirectional_LSTM else ""} LSTM with {lstm_units} '
                  f'units and a dense layer with output dimension {self.summary_dim} as summary network')

            summary_net = SplitNetwork(
                num_splits=2,
                split_data_configurator=split_data if self.n_obs_per_measure < 4 else split_data_2d,
                network_type=SequenceNetwork,
                network_kwargs=network_kwargs)

        elif self.summary_network_type == 'transformer':
            summary_net = TimeSeriesTransformer(
                input_dim=self.n_obs_per_measure,
                summary_dim=self.summary_dim,
                bidirectional=self.bidirectional_LSTM
            )
            print(f'using a TimeSeriesTransformer with a {"bidirectional" if self.bidirectional_LSTM else ""} LSTM '
                  f'template and output dimension {self.summary_dim} as summary network')

        elif self.summary_network_type == 'sequence':
            summary_net = SequenceNetwork(
                summary_dim=self.summary_dim,
                num_conv_layers=self.num_conv_layers,
                lstm_units=lstm_units,
                bidirectional=self.bidirectional_LSTM)
            print(
                f'using {self.num_conv_layers} layers of MultiConv1D, '
                f'a {"bidirectional" if self.bidirectional_LSTM else ""} LSTM with {lstm_units} '
                f'units and a dense layer with output dimension {self.summary_dim} as summary network')
        else:
            raise ValueError(f'Unknown summary network type {self.summary_network_type}')

        # inference network
        coupling_settings = {  # dict overwrites default settings from BayesFlow
            "num_dense": self.n_dense_layers_in_coupling,
            'dense_args': dict(activation='elu')  # standard is ReLU which might be faster
        }

        inference_net = InvertibleNetwork(num_params=self.n_params,
                                          num_coupling_layers=self.n_coupling_layers,
                                          coupling_design=self.coupling_design,
                                          coupling_settings=coupling_settings)
        print(f'using a {self.n_coupling_layers}-layer cINN as inference network '
              f'with {self.n_dense_layers_in_coupling} layers of design {self.coupling_design}')
        if self.latent_dist == 'normal':
            latent_dist = None
        elif self.latent_dist == 't-student':
            latent_dist = tfp.distributions.MultivariateStudentTLinearOperator(
                df=10,  # reasonable choice if one is unsure about the tail behavior
                loc=np.zeros(inference_net.latent_dim, dtype=np.float32),
                scale=tf.linalg.LinearOperatorDiag(np.ones(inference_net.latent_dim, dtype=np.float32)),
            )
        else:
            raise ValueError(f'Unknown latent distribution {self.latent_dist}')

        self.amortizer = AmortizedPosterior(inference_net, summary_net, latent_dist=latent_dist)
        return

    def _build_prior(self) -> None:
        """
        Build prior distribution.
        Returns: prior, configured_input - prior distribution and function to configure input

        """
        if self.prior_type == 'normal':
            print('Using normal prior')
            print('prior mean:', self.prior_mean)
            print('prior covariance diagonal:', self.prior_cov.diagonal())
            self.prior = Prior(batch_prior_fun=partial(batch_gaussian_prior,
                                                       mean=self.prior_mean,
                                                       cov=self.prior_cov),
                               param_names=self.log_param_names)
        elif self.prior_type == 'uniform':
            print('Using uniform prior')
            print(self.prior_bounds)
            self.prior = Prior(batch_prior_fun=partial(batch_uniform_prior,
                                                       prior_bounds=self.prior_bounds),
                               param_names=self.log_param_names)
            self.prior_mean = (np.diff(self.prior_bounds) / 2).flatten() + self.prior_bounds[:, 0]
            self.prior_std = (np.diff(self.prior_bounds) / 2).flatten() ** 2 / 12  # uniform variance
            self.prior_cov = np.diag(self.prior_std ** 2)
        else:
            raise ValueError('Unknown prior type')
        return

    def _reconfigure_samples(self, samples: np.ndarray) -> np.ndarray:
        """
        Reconfigure samples from the prior distribution to the original parameter space.
        Args:
            samples: np.ndarray - (#samples, #parameters)

        Returns: np.ndarray - (#samples, #parameters)

        """
        return self.prior_mean + samples * self.prior_std

    def draw_posterior_samples(self, data: Union[list, np.ndarray], n_samples: int) -> np.ndarray:
        """
        Function to draw samples from the posterior distribution.
        Takes care of different data formats and normalization.

        Args:
            data: [list, np.ndarray] - the data
            n_samples: int - number of samples to draw

        Returns: samples - np.ndarray - (#data, #samples, #parameters)

        """
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                # just data from one individual
                data = data[np.newaxis, :]
            posterior_draws = self.amortizer.sample({'summary_conditions': data}, n_samples=n_samples)
        else:
            # data is a list (different lengths, e.g. number of observations)
            input_list = []
            for d in data:
                # make bayesflow dict
                input_list.append({'summary_conditions': d[np.newaxis, :]})
            posterior_draws = self.amortizer.sample_loop(input_list, n_samples=n_samples)
            posterior_draws = posterior_draws.reshape((len(data), n_samples, self.n_params))
        return self._reconfigure_samples(posterior_draws)

    def generate_simulations_from_prior(self,
                                        n_samples: int,
                                        trainer: Trainer) -> dict:
        """
        Function to generate samples from the prior distribution.
        Takes care of different data formats and normalization.
        If changeable_obs_n is True, the number of observations might change per simulation.
        """
        if self.simulator is None:
            raise ValueError('Simulator not defined yet.')

        generative_model = GenerativeModel(prior=self.prior,
                                           simulator=self.simulator,
                                           skip_test=True)

        if self.changeable_obs_n:
            # sample separately for each data point since points may have different number of observations
            new_sims = {'summary_conditions': [], 'parameters': []}
            for i_sample in range(n_samples):
                single_new_sims = trainer.configurator(generative_model(1))
                single_new_sims['parameters'] = self._reconfigure_samples(single_new_sims['parameters'])
                new_sims['summary_conditions'].append(single_new_sims['summary_conditions'].squeeze(axis=0))
                new_sims['parameters'].append(single_new_sims['parameters'].squeeze(axis=0))
            new_sims['parameters'] = np.array(new_sims['parameters'])
        else:
            new_sims = trainer.configurator(generative_model(n_samples))
            new_sims['parameters'] = self._reconfigure_samples(new_sims['parameters'])
        return new_sims

    @abstractmethod
    def plot_example(self, params: Optional[np.ndarray] = None) -> None:
        """Plots an individual trajectory of an individual in this model."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def prepare_plotting(data: np.ndarray, params: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
        raise NotImplementedError


def batch_gaussian_prior(mean: np.ndarray,
                         cov: np.ndarray,
                         batch_size: int) -> np.ndarray:
    """
    Samples from the prior 'batch_size' times.
    ----------

    Arguments:h
    mean : np.ndarray - mean of the normal distribution
    cov: np.ndarray - covariance of the normal distribution
    batch_size : int - number of samples to draw from the prior
    ----------

    Output:
    p_samples : np.ndarray of shape (batch size, parameter dimension) -- the samples batch of parameters
    """
    prior_batch = np.random.multivariate_normal(mean=mean,
                                                cov=cov,
                                                size=batch_size)
    return prior_batch


def batch_uniform_prior(prior_bounds: np.ndarray,
                        batch_size: int) -> np.ndarray:
    """Sample from a uniform prior distribution."""
    prior_batch = np.random.uniform(low=prior_bounds[:, 0],
                                    high=prior_bounds[:, 1],
                                    size=(batch_size, prior_bounds.shape[0]))
    return prior_batch


def configure_input(forward_dict: dict,
                    prior_means: Optional[np.ndarray] = None,
                    prior_stds: Optional[np.ndarray] = None,
                    show_more: bool = False) -> dict:
    """
        Function to configure the simulated quantities (i.e., simulator outputs)
        into a neural network-friendly (BayesFlow) format.
    """

    # Prepare placeholder dict
    out_dict = {}

    # Convert data to float32
    data = forward_dict['sim_data']

    # simulate batch
    if data.ndim != 3:  # so not (batch_size, n_obs, n_measurements)
        # just a single parameter set
        if data.ndim == 1:
            data = data[np.newaxis, :, np.newaxis]
        elif data.ndim == 2:
            data = data[np.newaxis, :, :]
        else:
            raise ValueError(f'Invalid data dimension {data.ndim}')

    # Extract prior draws
    params = forward_dict['prior_draws']

    # z-standardize with previously computed means
    if prior_means is not None and prior_stds is not None:
        params = (params - prior_means) / prior_stds

    # Remove a batch if it contains nan, inf or -inf
    idx_keep = np.all(np.isfinite(data), axis=(1, 2))
    if not np.all(idx_keep):
        print(f'Invalid value(s) encountered...removing {idx_keep.size - np.sum(idx_keep)} entry(ies) from batch')
        if show_more:
            bad_params = np.exp(params[~idx_keep])
            for p in bad_params.T:
                plt.figure()
                plt.hist(p)
                plt.show()

    # Add to keys
    out_dict['summary_conditions'] = data[idx_keep].astype(np.float32)
    out_dict['parameters'] = params[idx_keep].astype(np.float32)
    return out_dict
