#!/usr/bin/env python
# coding: utf-8

# # Amortized Inference for the de Pillis NLME Model

import itertools
import os
import pathlib
from datetime import datetime
from typing import Optional, Union
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayesflow.simulation import Simulator
from juliacall import Main as jl
from juliacall import Pkg as jlPkg
from juliacall import convert as jlconvert

from inference.base_nlme_model import NlmeBaseAmortizer
from bayesflow.simulation import Prior

env = os.path.join(pathlib.Path(__file__).parent.resolve(), 'SimulatorDePillis')
jlPkg.activate(env)
jl.seval("using SimulatorDePillis")

path_to_data = '../data/dePillis/'

NAN_DOSING: float = 600.
FEATURES: str = ['old', 'wide', 'large'][1]
TIME_MEAN_NORM: float = 201.
TIME_STD_NORM: float = 100.
MEASUREMENTS_MEAN_NORM: float = 1200.
MEASUREMENTS_STD_NORM: float = 865.
INCLUDE_LIN_ERROR: bool = True



def measurement_model(y: np.ndarray, censoring: float = 2500., threshold: float = 0.001) -> np.ndarray:
    """
    Applies a measurement function to a given variable.

    Parameters:
    y (np.ndarray): Measurements of the variable to which the measurement function should be applied.
    censoring (float): Right-censoring value for the measurement function.
    threshold (float): Left-censoring value for the measurement function.

    Returns:
    np.ndarray: The result of applying the measurement function to the input variable.
    """
    y[y < threshold] = threshold
    y[y > censoring] = censoring
    return y


def prop_noise(y: np.ndarray, error_constant: float, error_prop: float) -> np.ndarray:
    """
    Proportional error model for given trajectories.

    Parameters:
    y (np.ndarray): The trajectory to which noise should be added.
    error_constant (float): The constant error term.
    error_prop (float): The proportional error term.

    Returns:
    np.ndarray: The noisy trajectories.
    """
    noise = np.random.normal(loc=0, scale=1, size=y.shape)
    return y + (error_constant + error_prop * y) * noise


def batch_simulator(
        param_batch: np.ndarray,
        t_measurements: Optional[np.ndarray] = None,
        t_doses: Optional[np.ndarray] = None,
        with_noise: bool = True,
        convert_to_bf_batch: bool = True,
) -> np.ndarray:
    """
    Simulate a batch of parameter sets.

    param_batch: np.ndarray - (#simulations, #parameters) or (#parameters)

    If time points for measurements and dosing events are not given, they are sampled.
    If convert_to_bf_batch is True, the output is in the format used by the bayesflow summary model, else only the
        measurements are returned.
    """
    # starting values
    x0 = np.array([0.0])
    resample_m_time_points, resample_d_time_points = False, False

    if t_measurements is not None:
        n_measurements = len(t_measurements)
    else:
        n_measurements = 4
        resample_m_time_points = True

    if t_doses is not None:
        n_doses = len(t_doses)
    else:
        n_doses = 3
        resample_d_time_points = True

    # convert to julia types
    jl_x0 = jlconvert(jl.Vector[jl.Float64], x0)

    # simulate batch
    if param_batch.ndim == 1:  # so not (batch_size, params)
        # just a single parameter set
        param_batch = param_batch[np.newaxis, :]
    n_sim = param_batch.shape[0]
    if convert_to_bf_batch:
        # create output batch containing all information for bayesflow summary model
        if FEATURES == 'old':
            output_batch = np.zeros((n_sim, n_measurements+n_doses, 3), dtype=np.float32)
        elif FEATURES == 'large':
            output_batch = np.zeros((n_sim, n_measurements, 5), dtype=np.float32)
        elif FEATURES == 'wide':
            output_batch = np.zeros((n_sim, n_measurements+n_doses, 6), dtype=np.float32)
        else:
            raise ValueError('Unknown feature format')
    else:
        # just return the simulated data
        output_batch = np.zeros((n_sim, n_measurements))

    for pars_i, log_params in enumerate(param_batch):
        params = np.exp(log_params)
        infected_before_first_dose = 1  # patient is infected before the first dose
        if params[0] < 0.1:
            infected_before_first_dose = 0

        # sample the measurement and dosing time points
        if resample_m_time_points and resample_d_time_points:
            t_measurements, t_doses = get_time_points()
        else:
            if resample_m_time_points:
                t_measurements = get_measurement_time_points()
            elif resample_d_time_points:
                t_doses = get_dosing_time_points()

        # set the first parameter to 0 if the patient was not infected before the first dose
        if infected_before_first_dose == 0:
            params[0] = 0.0

        # convert to julia types
        if INCLUDE_LIN_ERROR:
            jl_parameter = jlconvert(jl.Vector[jl.Float64], params[:-2])
        else:
            jl_parameter = jlconvert(jl.Vector[jl.Float64], params[:-1])
        jl_dosetimes = jlconvert(jl.Vector[jl.Float64], t_doses)
        jl_t_measurement = jlconvert(jl.Vector[jl.Float64], t_measurements)

        # simulate
        y_sim = jl.simulateDePillis(jl_parameter,
                                    jl_x0,
                                    jl_dosetimes,
                                    jl_t_measurement).to_numpy()
        # we only observe the antibody concentration
        y_sim = y_sim[1]

        # apply noise
        if with_noise:
            if INCLUDE_LIN_ERROR:
                y_sim = prop_noise(y_sim, error_constant=params[-2], error_prop=params[-1])
            else:
                y_sim = prop_noise(y_sim, error_constant=0, error_prop=params[-1])

        # applying censoring
        y_sim = measurement_model(y_sim)

        # reshape the data to fit in one numpy array
        if convert_to_bf_batch:
            if FEATURES == 'old':
                output_batch[pars_i, :, :] = convert_to_bf_format_old(
                    y=y_sim,
                    t_measurements=t_measurements,
                    infected_before_first_dose=infected_before_first_dose,
                    doses_time_points=t_doses
                )
            elif FEATURES == 'large':
                output_batch[pars_i, :, :] = convert_to_bf_format_large(
                    y=y_sim,
                    t_measurements=t_measurements,
                    infected_before_first_dose=infected_before_first_dose,
                    doses_time_points=t_doses
                )
            elif FEATURES == 'wide':
                output_batch[pars_i, :, :] = convert_to_bf_format(
                    y=y_sim,
                    t_measurements=t_measurements,
                    infected_before_first_dose=infected_before_first_dose,
                    doses_time_points=t_doses
                )
            else:
                raise ValueError('Unknown feature format')
        else:
            output_batch[pars_i, :] = y_sim

    if n_sim == 1:
        # remove batch dimension
        return output_batch[0]
    return output_batch


def batch_gaussian_prior_de_pillis(
        mean: np.ndarray,
        cov: np.ndarray,
        batch_size: int,
        rand_infected_before_first_dose: float = 0.2
) -> np.ndarray:
    """
        Samples from the prior 'batch_size' times.
        ----------

        Arguments:h
        mean : np.ndarray - mean of the normal distribution
        cov: np.ndarray - covariance of the normal distribution
        batch_size : int - number of samples to draw from the prior
        rand_infected_before_first_dose : float - probability of being infected before the first dose,
                                         if not infected changes the prior
        ----------

        Output:
        p_samples : np.ndarray of shape (batch size, parameter dimension) -- the samples batch of parameters
        """
    prior_batch = np.random.multivariate_normal(mean=mean,
                                                cov=cov,
                                                size=batch_size)
    # set the first parameter to 0 if the patient was not infected before the first dose
    prior_batch[np.random.uniform(0, 1, size=batch_size) > rand_infected_before_first_dose, 0] = np.log(0.01)
    return prior_batch


def simulate_single_patient(
        param_batch: np.ndarray,
        patient_data: np.ndarray,
        full_trajectory: bool = False,
        with_noise: bool = False,
) -> np.ndarray:
    """uses the batch simulator to simulate a single patient"""
    y, t_measurements, doses_time_points, _ = convert_bf_to_observables(patient_data)
    if full_trajectory:
        t_measurements = np.linspace(0, t_measurements[-1], 100)

    y_sim = batch_simulator(param_batch,
                            t_measurements=t_measurements,
                            t_doses=doses_time_points,
                            with_noise=with_noise,
                            convert_to_bf_batch=False)
    return y_sim


def convert_to_bf_format_old(
        y: np.ndarray,
        t_measurements: np.ndarray,
        doses_time_points: np.ndarray,
        infected_before_first_dose: int,
) -> np.ndarray:
    """
    converts all data to the format used by the bayesflow summary model
        (y_transformed, timepoints_transformed, 0) concatenated with
        (infected_before_first_dose, dosing_time_transformed / scaling_time, 1)
    and then sort by time
    """
    y_transformed = (y - MEASUREMENTS_MEAN_NORM) / MEASUREMENTS_STD_NORM
    t_measurements_transformed = (t_measurements - TIME_MEAN_NORM) / TIME_STD_NORM
    doses_time_points_transformed = (doses_time_points - TIME_MEAN_NORM) / TIME_STD_NORM

    # reshape the data to fit in one numpy array
    measurements = np.stack((y_transformed,
                             t_measurements_transformed,
                             np.zeros_like(t_measurements)),
                            axis=1)
    doses = np.stack((np.ones_like(doses_time_points) * infected_before_first_dose,
                      doses_time_points_transformed,
                      np.ones_like(doses_time_points)),
                     axis=1)
    bf_format = np.concatenate((measurements, doses), axis=0)
    bf_format_sorted = bf_format[bf_format[:, 1].argsort()]
    return bf_format_sorted.astype(np.float32)


def convert_to_bf_format(
        y: np.ndarray,
        t_measurements: np.ndarray,
        doses_time_points: np.ndarray,
        infected_before_first_dose: int,
) -> np.ndarray:
    """
    converts all data to the format used by the bayesflow summary model

    A feature vector is constructed for each measurement, order by time, including:
        Normalized measurement value (log10) or 0 if dosing event.
        Binary flag indicating if it is a measurement (0) or a dosing event (1).
        Number of doses administered before the measurement.
        The flag 'infected_before_first_dose'.
        Time since the last dose. (log10)
        Normalized time of measurement or dosing event. (log10)
    """
    # reshape the data to fit in one numpy array
    measurements_features = np.stack((
        np.log10(y),
        np.zeros_like(t_measurements),
        np.ones_like(t_measurements) * np.nan,  # number of doses: will be filled later
        np.ones_like(t_measurements) * infected_before_first_dose,
        np.ones_like(t_measurements) * np.nan,  # time since last dose: will be filled later,
        np.log10(t_measurements)
    ), axis=1)
    doses_features = np.stack((
        np.zeros_like(doses_time_points),  # no dosing amounts
        np.ones_like(doses_time_points),
        np.array([1., 2., 3.]),  # number of doses
        np.ones_like(doses_time_points) * infected_before_first_dose,
        (doses_time_points - 2.) / NAN_DOSING,  # first dose is always at 2
        np.log10(doses_time_points),
    ), axis=1)

    bf_format = np.concatenate((measurements_features, doses_features), axis=0)
    bf_format_sorted = bf_format[bf_format[:, -1].argsort()]

    # fill in the number of doses and time since last dose
    for i in range(1, len(bf_format_sorted)):
        if np.isnan(bf_format_sorted[i, 2]):
            # replace nan by value of previous row
            bf_format_sorted[i, 2] = bf_format_sorted[i - 1, 2]
            # get time of last dose
            for j in range(i - 1, -1, -1):
                if bf_format_sorted[j, 1] == 1:  # dosing event
                    time_since_dose = np.power(10, bf_format_sorted[i, 5]) - np.power(10, bf_format_sorted[j, 5])
                    bf_format_sorted[i, 4] = time_since_dose / NAN_DOSING
                    break
    return bf_format_sorted.astype(np.float32)


def convert_to_bf_format_large(
        y: np.ndarray,
        t_measurements: np.ndarray,
        doses_time_points: np.ndarray,
        infected_before_first_dose: int,
) -> np.ndarray:
    """
    converts all data to the format used by the bayesflow summary model

    A feature vector is constructed for each measurement including:
        Normalized measurement value.
        Normalized time of measurement.
        Time since the last dose.
        Number of doses administered before the measurement (excluding one dose)
        The flag 'infected_before_first_dose'.
    """
    # Normalize measurements and time points
    y_normalized = (y - MEASUREMENTS_MEAN_NORM) / MEASUREMENTS_STD_NORM
    t_measurements_normalized = (t_measurements - TIME_MEAN_NORM) / TIME_STD_NORM
    doses_time_points_normalized = (doses_time_points - TIME_MEAN_NORM) / TIME_STD_NORM

    # Initialize an empty list to store feature vectors
    feature_vectors = []

    # Compute the features for each measurement
    for i, (measurement, t_measurement) in enumerate(zip(y_normalized, t_measurements_normalized)):
        # Time since last dose
        if t_measurement < doses_time_points_normalized[0]:
            raise ValueError('Measurement time is before the first dose')
        else:
            # Find the index of the last dose before the current measurement time
            last_dose_index = np.searchsorted(doses_time_points_normalized, t_measurement, side='right') - 1
            time_since_last_dose = t_measurement - doses_time_points_normalized[last_dose_index]
            num_doses = last_dose_index + 1  # +1 because doses are 1-indexed

        # Create the feature vector for the current measurement
        feature_vector = np.array([
            measurement,
            t_measurement,
            time_since_last_dose,
            num_doses,
            infected_before_first_dose
        ])

        # Append the feature vector to the list
        feature_vectors.append(feature_vector)

    # Convert the list of feature vectors to a numpy array
    feature_matrix = np.vstack(feature_vectors).astype(np.float32)
    return feature_matrix


def convert_bf_to_observables_old(
        output: np.ndarray
) -> (np.ndarray, np.ndarray, float, float):
    """
    converts data in the bayesflow summary model format back to observables
        (y, timepoints / scaling_time, 0) concatenated with (infected_before_first_dose, timepoints / scaling_time, 1)
    """
    measurements = output[np.where(output[:, 2] == 0)]
    y = measurements[:, 0] * MEASUREMENTS_STD_NORM + MEASUREMENTS_MEAN_NORM
    t_measurements = measurements[:, 1] * TIME_STD_NORM + TIME_MEAN_NORM

    doses = output[np.where(output[:, 2] == 1)]
    infected_before_first_dose = doses[0, 0]
    doses_time_points = doses[:, 1] * TIME_STD_NORM + TIME_MEAN_NORM
    return y, t_measurements, doses_time_points, infected_before_first_dose


def convert_bf_to_observables(
        output: np.ndarray
) -> (np.ndarray, np.ndarray, float, float):
    """
    converts data in the bayesflow summary model format back to observables

    A feature vector for each measurement, order by time, includes:
        Normalized measurement value (log10) or 0 if dosing event.
        Binary flag indicating if it is a measurement (0) or a dosing event (1).
        Number of doses administered before the measurement.
        The flag 'infected_before_first_dose'.
        Time since the last dose. (log10)
        Normalized time of measurement or dosing event. (log10)

    Returns:
        y: measurement values
        t_measurements: time of measurements
        doses_time_points: time of dosing events
        infected_before_first_dose: binary flag if infected before first dose
    """
    measurements = output[np.where(output[:, 1] == 0)]
    y = np.power(10, measurements[:, 0])
    t_measurements = np.power(10, measurements[:, -1])

    doses = output[np.where(output[:, 1] == 1)]
    doses_time_points = np.power(10, doses[:, -1])

    infected_before_first_dose = output[0, 3]  # infected_before_first_dose is the same for observations
    return y, t_measurements, doses_time_points, infected_before_first_dose


def convert_bf_large_to_observables(
        output: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, float):
    """
    Converts data in the BayesFlow summary model format back to observables.

    The feature vector given is for each measurement:
        Normalized measurement value.
        Normalized time of measurement.
        Time since the last dose.
        Number of doses administered before the measurement.
        The flag 'infected_before_first_dose'.
    """
    # Extracting columns from the output matrix
    measurements_normalized = output[:, 0]
    t_measurements_normalized = output[:, 1]
    time_since_last_dose_normalized = output[:, 2]
    num_doses = output[:, 3]
    infected_before_first_dose = output[0, 4]  # Assuming this is the same for all measurements
    
    # Reverting normalization for measurements and time points
    y = measurements_normalized * MEASUREMENTS_STD_NORM + MEASUREMENTS_MEAN_NORM
    t_measurements = t_measurements_normalized * TIME_STD_NORM + TIME_MEAN_NORM

    # Initialize doses_time_points with -1 to identify missing doses
    doses_time_points = np.ones(3) * NAN_DOSING
    # add first dose (might not be possible to reconstruct from features)
    doses_time_points[0] = 2  # assumes that this is the same for all patients (which is true)
    
    # Reconstruct doses time points based on the time_since_last_dose information
    for idx, dose_count in enumerate(num_doses):
        dose_count = int(dose_count.item())
        if dose_count > 0:  # Only consider entries with doses
            dose_index = dose_count - 1  # Doses are 1-indexed in the feature vectors
            if doses_time_points[dose_index] == NAN_DOSING:  # If not yet set, compute the dose time
                doses_time_points_normalized = t_measurements_normalized[idx] - time_since_last_dose_normalized[idx]
                doses_time_points[dose_index] = int(doses_time_points_normalized * TIME_STD_NORM + TIME_MEAN_NORM)
            else:
                # check entry if it is the same
                doses_time_points_normalized = t_measurements_normalized[idx] - time_since_last_dose_normalized[idx]
                doses_time_points_test = int(doses_time_points_normalized * TIME_STD_NORM + TIME_MEAN_NORM)
                if doses_time_points[dose_index] != doses_time_points_test:
                    print('Warning: Dose time points are not the same for the same dose number:',
                          doses_time_points[dose_index], doses_time_points_test)
    
    return y, t_measurements, doses_time_points, infected_before_first_dose


class dePillisModel(NlmeBaseAmortizer):
    def __init__(self, name: str = 'dePillisModel', network_idx: int = -1, load_best: bool = False,
                 prior_type: str = 'normal',  # normal or uniform
                 ):
        # define names of parameters
        param_names = ['Ab0', 'r1', 'r2', 'r3', 'r4', 'k1', 'k2',
                       'error_lin', 'error_prop']

        # define prior values (for log-parameters)
        if INCLUDE_LIN_ERROR:
            prior_mean = np.log([10, 0.01, 0.5, 0.00001, 0.00001, 10.0, 55.0, 0.1, 0.1])
            prior_cov = np.diag(np.array([5., 5., 5., 5., 5., 3., 3., 1., 0.6]))  # todo: changed noise from 1.
        else:
            prior_mean = np.log([10, 0.01, 0.5, 0.00001, 0.00001, 10.0, 55.0, 0.1])
            prior_cov = np.diag(np.array([5., 5., 5., 5., 5., 3., 3., 0.6]))

        # define prior bounds for uniform prior
        # self.prior_bounds = np.array([[-10, 5], [-5, 10], [-5, 10], [-20, 0], [-10, 0], [-10, 0], [-10, 0]])
        #self.prior_bounds = np.array([[-5, 7], [-5, 7], [-5, 7], [-5, 7], [-5, 0], [-5, 0], [-5, 0]])

        super().__init__(name=name,
                         network_idx=network_idx,
                         load_best=load_best,
                         param_names=param_names,
                         prior_mean=prior_mean,
                         prior_cov=prior_cov,
                         prior_type=prior_type,
                         max_n_obs=7)  # 4 measurements, 3 doses

        # define simulator
        self.simulator = Simulator(batch_simulator_fun=batch_simulator)
        print(f'Using the model {name}')

    def _build_prior(self) -> None:
        """
        Build prior distribution.
        Returns: prior, configured_input - prior distribution and function to configure input

        """
        print('Using normal prior adapted for dePillis')
        print('prior mean:', self.prior_mean)
        print('prior covariance diagonal:', self.prior_cov.diagonal())
        self.prior = Prior(batch_prior_fun=partial(batch_gaussian_prior_de_pillis,
                                                   mean=self.prior_mean,
                                                   cov=self.prior_cov),
                           param_names=self.log_param_names)
        return

    def load_amortizer_configuration(self, model_idx: int = 0, load_best: bool = False) -> str:
        self.n_epochs = 100 # 1000
        self.summary_dim = self.n_params * 2
        if FEATURES == 'old':
            self.n_obs_per_measure = 3  # time and measurement + event type (measurement = 0, dosing = 1)
        elif FEATURES == 'large':
            self.n_obs_per_measure = 5
        elif FEATURES == 'wide':
            self.n_obs_per_measure = 6
        else:
            raise ValueError('Unknown feature format')

        # load best
        if load_best:
            model_idx = 3  # 2

        # wide features
        bidirectional_LSTM = [True]
        n_coupling_layers = [7]  # 7 seems to work best
        n_dense_layers_in_coupling = [2, 3]
        num_conv_layers = [2]
        coupling_design = ['spline']
        summary_network_type = ['sequence', 'transformer']
        latent_dist = ['normal', 't-student']

        # 5 better than 3, but 3 seems to generalize better
        # 1000 epochs
        # 0: -0.1846 (999 epochs, Bi_LSTM, 2, normal)
        # 1: 3.5279 (244 epochs, Bi_LSTM, 2, t-student)
        # 2: 2.8107 (218 epochs, transformer, 2, normal)
        # 3: -0.1990 (995 epochs, transformer, 2, t-student)
        # 4: 7.4811 (273 epochs, Bi_LSTM, 3, normal)
        # 5: -0.2347 (996 epochs, Bi_LSTM, 3, t-student)
        # 6: -0.0448 (980 epochs, transformer, 3, normal)
        # 7:  0.0714 (947 epochs, transformer, 3, t-student)

        # 500 epochs
        # 0: 1.5593 (496 epochs, Bi_LSTM, 2, normal)
        # 1: -0.6357 (496 epochs, Bi_LSTM, 2, t-student)
        # 2: 1.2162 (367 epochs, transformer, 2, normal)
        # 3: 0.4292 (498 epochs, transformer, 2, t-student)
        # 4: 2.0496 (498 epochs, Bi_LSTM, 3, normal)
        # 5: -0.2481 (494 epochs, Bi_LSTM, 3, t-student)
        # 6: 2.0370 (490 epochs, transformer, 3, normal)
        # 7: 0.1666 (485 epochs, transformer, 3, t-student)

        # 100 epochs with MMD
        # 0: 0.3130 (98 epochs, Bi_LSTM, 2, normal)
        # 1: 0.7492 (97 epochs, Bi_LSTM, 2, t-student)
        # 2: 2.1403 (98 epochs, transformer, 2, normal)
        # 3: 2.2115 (97 epochs, transformer, 2, t-student)
        # 4: 0.5614 (99 epochs, Bi_LSTM, 3, normal)
        # 5: 0.7120 (98 epochs, Bi_LSTM, 3, t-student)
        # 6: 2.0510 (98 epochs, transformer, 3, normal)
        # 7: 2.5470 (97 epochs, transformer, 3, t-student)


        # large features
        # bidirectional_LSTM = [False, True]
        # n_coupling_layers = [6, 7, 8]  # 7 seems to work best
        # n_dense_layers_in_coupling = [2, 3]
        # num_conv_layers = [0, 2]  # conv improves results
        # coupling_design = ['spline']
        # summary_network_type = ['sequence']
        # latent_dist = ['normal']

        # 0: -0.8680 (900 epochs, early stopped, LSTM, 0 conv, 6-2)
        # 1: -1.9661 (994 epochs, LSTM, 2 conv, 6-2)
        # 2: -1.4744 (992 epochs, early stopped, LSTM, 0 conv, 6-3)
        # 3: -1.6887 (996 epochs, LSTM, 2 conv, 6-3)
        # 4: -1.4306 (980 epochs, early stopped, LSTM, 0 conv, 7-2)
        # 5: -0.5700 (993 epochs, early stopped, LSTM, 2 conv, 7-2)
        # 6: 1.5495 (677 epochs, early stopped, LSTM, 0 conv, 7-3)
        # 7: -1.9119 (994 epochs, LSTM, 2 conv, 7-3)
        # 8: -0.6206 (996 epochs, LSTM, 0 conv, 8-2)
        # 9: -1.7018 (987 epochs, early stopped, LSTM, 2 conv, 8-2)
        # 10: 0.6066 (833 epochs, early stopped, LSTM, 0 conv, 8-3)
        # 11: -0.6964 (993 epochs, early stopped, LSTM, 2 conv, 8-3)
        # 12: -1.2352 (991 epochs, early stopped, bi-LSTM, 0 conv, 6-2)
        # 13: 3.2452 (181 epochs, early stopped, bi-LSTM, 2 conv, 6-2)
        # 14: 3.2683 (112 epochs, early stopped, bi-LSTM, 0 conv, 6-3)
        # 15: -1.9943 (995 epochs, bi-LSTM, 2 conv, 6-3)
        # 16: 1.0736 (652 epochs, early stopped, bi-LSTM, 0 conv, 7-2)
        # 17: -1.1917 (953 epochs, early stopped, bi-LSTM, 2 conv, 7-2)
        # 18: -0.5859 (999 epochs, bi-LSTM, 0 conv, 7-3)
        # 19: -2.0253 (985 epochs, early stopped, bi-LSTM, 2 conv, 7-3)
        # 20: 1.9376 (404 epochs,  early stopped, bi-LSTM, 0 conv, 8-2)
        # 21: -1.9535 (998 epochs, bi-LSTM, 2 conv, 8-2)
        # 22: 0.5300 (808 epochs, early stopped, bi-LSTM, 0 conv, 8-3)
        # 23: -1.2211 (996 epochs, bi-LSTM, 2 conv, 8-3)

        # old features
        # transformer
        # bidirectional_LSTM = [False]
        # n_coupling_layers = [7]
        # n_dense_layers_in_coupling = [2, 3]
        # coupling_design = ['spline']
        # summary_network_type = ['transformer']
        # latent_dist = ['normal', 't-student']

        # transformer
        # 0: 1.6037
        # 1: -0.8997
        # 2: -0.5014
        # 3: -0.9977

        # 1000  (bidirectional LSTM on/off, added 8 coupling layers)
        # 0: -1.6469 (996 epochs, normal, early stopped) # lstm, 6 layers, 2
        # 1: 0.5922 (734 epochs, normal, early stopped) # lstm, 6 layers, 3
        # 2: -2.5560 (988 epochs, normal, early stopped)  #lstm, 7 layers,2
        # 3: -1.8822 (998 epochs, normal) # lstm, 7 layers, 3
        # 4: -2.4473 (997 epochs, normal)  #lstm, 8 layers,2
        # 5:  3.1764 (203 epochs, normal, failed)  # lstm, 8 layers, 3
        # 6: -2.4837 (959 epochs, normal)  # bi-lstm, 6 layers, 2
        # 7: -0.9242 (996 epochs, normal)  # bi-lstm, 6 layers, 3
        # 8: -2.0800 (991 epochs, normal, early stopped)  # bi-lstm, 7 layers, 2
        # 9: -2.4642 (993 epochs, normal, early stopped)  # bi-lstm, 7 layers, 3
        # 10: -1.4012 (925 epochs, normal early stopped)  # bi-lstm, 8 layers, 2
        # 11: -2.2700 (994 epochs, normal)  # bi-lstm, 8 layers, 3

        #bidirectional_LSTM = [False, True]
        #n_coupling_layers = [6, 7, 8]
        #n_dense_layers_in_coupling = [2, 3]
        #num_conv_layers = [2]
        #coupling_design = ['spline']
        #summary_network_type = ['sequence']
        #latent_dist = ['normal']

        # 750  (added t-student distribution)
        # 0: -2.1249 (746 epochs, normal)
        # 1: -1.5960 (745 epochs, t-student)
        # 2: -1.9177 (749 epochs, normal)
        # 3: -1.4752 (744 epochs, t-student)
        # 4: -1.7052 (749 epochs, normal)
        # 5: -1.6547 (742 epochs, t-student)
        # 6: -1.7814 (742 epochs, normal)
        # 7: -0.9790 (746 epochs, t-student)

        # bidirectional_LSTM = [True]
        # n_coupling_layers = [6, 7]
        # n_dense_layers_in_coupling = [2, 3]
        # coupling_design = ['spline']
        # summary_network_type = ['sequence']
        # latent_dist = ['normal', 't-student']

        # 500
        # 0: -1.4427 (495 epochs)
        # 1: -1.1849 (494 epochs)
        # 2: -1.2068 (498 epochs)
        # 3: 1.9243 (275 epochs) failed

        # bidirectional_LSTM = [True]
        # n_coupling_layers = [6, 7]
        # n_dense_layers_in_coupling = [2, 3]
        # coupling_design = ['spline']
        # summary_network_type = ['sequence']
        # latent_dist = ['normal']

        combinations = list(itertools.product(bidirectional_LSTM, n_coupling_layers,
                                              n_dense_layers_in_coupling, num_conv_layers,
                                              coupling_design, summary_network_type,
                                              latent_dist))

        if model_idx >= len(combinations) or model_idx < 0:
            model_name = f'amortizer-dePillis-{self.prior_type}' \
                         f'-{self.summary_network_type}-summary' \
                         f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                         f'-{self.num_conv_layers}-conv-layers' \
                         f'-{self.n_coupling_layers}layers' \
                         f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                         f'-{self.latent_dist}' \
                         f'-{self.n_epochs}epochs' \
                         f'-{FEATURES}' \
                         f'-{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
            return model_name

        (self.bidirectional_LSTM,
         self.n_coupling_layers,
         self.n_dense_layers_in_coupling,
         self.num_conv_layers,
         self.coupling_design,
         self.summary_network_type,
         self.latent_dist) = combinations[model_idx]

        model_name = f'amortizer-dePillis-{self.prior_type}' \
                     f'-{self.summary_network_type}-summary' \
                     f'-{"Bi-LSTM" if self.bidirectional_LSTM else "LSTM"}' \
                     f'-{self.num_conv_layers}-conv-layers' \
                     f'-{self.n_coupling_layers}layers' \
                     f'-{self.n_dense_layers_in_coupling}coupling-{self.coupling_design}' \
                     f'-{self.latent_dist}' \
                     f'-{self.n_epochs}epochs' \
                     f'-{FEATURES}'
        return model_name

    def load_data(self,
                  n_data: Optional[int] = None,
                  load_covariates: bool = False,
                  synthetic: bool = False,
                  return_synthetic_params: bool = False,
                  seed: int = 0
                  ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if synthetic:
            assert isinstance(n_data, int)
            np.random.seed(seed)
            # Ab0, r1, r2, r3, r4, k1, k2, error_lin, error_prop
            if INCLUDE_LIN_ERROR:
                synthetic_mean = np.array([10, 0.01, 0.5, 0.00001, 0.00001, 10.0, 55.0, 0.1, 0.1])
            else:
                synthetic_mean = np.array([10, 0.01, 0.5, 0.00001, 0.00001, 10.0, 55.0, 0.1])
            synthetic_mean = np.random.normal(synthetic_mean, 1)
            synthetic_mean[synthetic_mean < 0.00001] = 0.00005
            synthetic_mean[[3, 4]] = 0.00001  # otherwise too large
            synthetic_mean[6] = 55.0  # k2 is fixed
            synthetic_mean[-1] = 0.05  # otherwise too large
            synthetic_mean = np.log(synthetic_mean)
            synthetic_cov = np.random.uniform(0.001, 1.5, size=self.n_params)
            synthetic_cov[-1] = 0.001  # error parameter
            synthetic_cov = np.diag(synthetic_cov)  # no fixed parameters
            c_params = np.array([0.5, 0.5])  # age, male

            # create individual parameters
            params = batch_gaussian_prior_de_pillis(mean=synthetic_mean,
                                                    cov=synthetic_cov,
                                                    batch_size=n_data)

            # calculate the true mean and std with respect to the covariates in Ab0
            true_mean = np.mean(params, axis=0)
            true_mean[0] = np.mean(params[params[:, 0] == np.log(0.01), 0])
            true_std = np.std(params, axis=0)
            true_std[0] = np.std(params[params[:, 0] == np.log(0.01), 0])
            true_params = np.concatenate((true_mean, true_std))

            # add covariates
            covariates = np.zeros((n_data, 3))
            # sample age from real ones
            df_measurements = pd.read_csv(path_to_data + "bf_data_measurements.csv", index_col=0)
            df_measurements['gender_code'] = df_measurements['gender'].astype('category').cat.codes
            # gender: 0 is first appearing gender in df, here female
            df_measurements['age_standardized'] = np.log(df_measurements['age'])
            covariates[:, 1] = np.random.choice(df_measurements['age_standardized'].values, n_data)
            covariates[:, 2] = np.random.choice(df_measurements['gender_code'].values, n_data)

            # add covariates to the parameters
            no_prev_infect = params[:, 0] == np.log(0.01)
            covariates[no_prev_infect, 0] = 0
            covariates[~no_prev_infect, 0] = 1
            params += c_params[0] * covariates[:, 1][:, np.newaxis]
            params += c_params[1] * covariates[:, 2][:, np.newaxis]
            params[no_prev_infect, 0] = np.log(0.01)

            patients_data = batch_simulator(param_batch=params)
            for i, o in enumerate(patients_data):
                if (o[o[:, -1] == 1, 0] == 1).any():
                    # previous infected
                    if covariates[i, 0] != 1:
                        raise ValueError('Previous infected but not in covariates')

            if return_synthetic_params and not load_covariates:
                return patients_data, true_params
            elif return_synthetic_params and load_covariates:
                return patients_data, covariates, true_params
            elif load_covariates:
                return patients_data, covariates
            return patients_data

        df_dosages = pd.read_csv(path_to_data + "bf_data_dosages_unique.csv")
        df_dosages.set_index('id_hcw', inplace=True)  # only now, so duplicates are removed correctly

        df_measurements = pd.read_csv(path_to_data + "bf_data_measurements.csv", index_col=0)
        df_measurements['gender_code'] = df_measurements['gender'].astype('category').cat.codes
        # gender: 0 is first appearing gender in df, here female
        df_measurements['age_standardized'] = np.log(df_measurements['age'])

        patients_data = []
        patients_covariates = []
        for p_id, patient in enumerate(df_measurements.index.unique()):
            m_times = df_measurements.loc[patient, ['days_after_first_dose']].values.flatten().astype(np.float32)
            y = df_measurements.loc[patient, ['res_serology']].values.flatten().astype(np.float32)
            infection_bef_1st_dose = df_measurements.loc[patient, ['infection_bef_1st_dose']].values.flatten()[0]
            d_times = df_dosages.loc[patient, ['day_1dose', 'day_2dose', 'day_3dose']].values.astype(np.float32)
            d_times[np.isnan(d_times)] = NAN_DOSING
            covariates = df_measurements.loc[patient, ['infection_bef_1st_dose', 'age_standardized', 'gender_code']].values[0].flatten()

            # apply measurement model
            y = measurement_model(y)
            if FEATURES == 'old':
                data = convert_to_bf_format_old(
                    y=y,
                    t_measurements=m_times,
                    infected_before_first_dose=infection_bef_1st_dose,
                    doses_time_points=d_times
                )
            elif FEATURES == 'large':
                data = convert_to_bf_format_large(
                    y=y,
                    t_measurements=m_times,
                    infected_before_first_dose=infection_bef_1st_dose,
                    doses_time_points=d_times
                )
            elif FEATURES == 'wide':
                data = convert_to_bf_format(
                    y=y,
                    t_measurements=m_times,
                    infected_before_first_dose=infection_bef_1st_dose,
                    doses_time_points=d_times
                )
            else:
                raise ValueError('Unknown feature type')
            if np.isnan(data).any():
                print(f'Patient {patient} has nan values and is removed.')
                # remove patients with nan values
                continue
            patients_data.append(data)
            patients_covariates.append(covariates)

            if n_data is not None and len(patients_data) == n_data:
                break

        if load_covariates:
            return np.stack(patients_data, axis=0), np.stack(patients_covariates, axis=0)
        return np.stack(patients_data, axis=0)

    def plot_example(self, params: Optional[np.ndarray] = None) -> None:
        """Plots an individual trajectory of an individual in this model."""
        if params is None:
            params = self.prior(1)['prior_draws'][0]

        output = batch_simulator(params)
        _ = self.prepare_plotting(output, params)

        plt.title(f'Patient Simulation')
        plt.legend()
        plt.show()
        return

    @staticmethod
    def prepare_plotting(data: np.ndarray, params: np.ndarray, ax: Optional[plt.Axes] = None,
                         with_noise: bool = False) -> plt.Axes:
        # convert BayesFlow format to observables
        if FEATURES == 'old':
            y, t_measurements, doses_time_points, infected_before_first_dose = convert_bf_to_observables_old(data)
        elif FEATURES == 'large':
            y, t_measurements, doses_time_points, infected_before_first_dose = convert_bf_large_to_observables(data)
        elif FEATURES == 'wide':
            y, t_measurements, doses_time_points, infected_before_first_dose = convert_bf_to_observables(data)
        else:
            raise ValueError('Unknown feature type')
        t_measurement_full = np.linspace(0, t_measurements[-1] + 100, 100)

        # simulate data
        sim_data = batch_simulator(param_batch=params,
                                   t_measurements=t_measurement_full,
                                   t_doses=doses_time_points,
                                   with_noise=with_noise,
                                   convert_to_bf_batch=False)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True)

        if len(params.shape) == 1:  # so not (batch_size, params)
            # just a single parameter set
            # plot simulated data
            ax.plot(t_measurement_full, sim_data, 'b', label='simulation')
        else:
            # calculate median and quantiles
            y_median = np.median(sim_data, axis=0)
            y_quantiles = np.percentile(sim_data, [2.5, 97.5], axis=0)

            # plot simulated data
            ax.fill_between(t_measurement_full, y_quantiles[0], y_quantiles[1],
                            alpha=0.2, color='orange', label='95% quantiles')
            ax.plot(t_measurement_full, y_median, 'b', label='median')

        # plot observed data
        if infected_before_first_dose == 0:
            ax.scatter(t_measurements, y, color='b', label='measurements')
        else:
            ax.scatter(t_measurements, y, color='b', label='measurements (infected before first dose)')

        # plot dosing events
        ax.vlines(doses_time_points, 0, 2500,
                  color='grey', alpha=0.5, label='dosing events')

        # plot censoring
        ax.hlines(2500, xmin=0, xmax=t_measurement_full[-1], linestyles='--',
                  color='green', label='censoring')

        ax.set_xlabel('Time (in days)')
        ax.set_ylabel('Measurements')
        return ax


def get_time_points() -> (np.ndarray, np.ndarray):
    """sample the time points for dosing and measurements"""
    df_measurements = pd.read_csv(path_to_data + 'bf_data_measurements.csv')
    patient_id = df_measurements.sample(1)['id_hcw'].values
    df_patient = df_measurements.loc[df_measurements['id_hcw'].values == patient_id]
    t_measurements = df_patient['days_after_first_dose'].values.flatten().astype(np.float32)

    df_dosage = pd.read_csv(path_to_data + 'bf_data_dosages_unique.csv')
    df_patient = df_dosage.loc[df_dosage['id_hcw'].values == patient_id]
    dosage = df_patient[['day_1dose', 'day_2dose', 'day_3dose']].values.flatten().astype(np.float32)

    # convert nan into maximal value
    dosage[np.isnan(dosage)] = NAN_DOSING

    return t_measurements, dosage


def get_measurement_time_points() -> np.ndarray:
    """sample the measurement time points"""
    df_measurements = pd.read_csv(path_to_data + 'bf_data_measurements.csv')
    df_patient = df_measurements.loc[df_measurements['id_hcw'].values == df_measurements.sample(1)['id_hcw'].values]
    t_measurements = df_patient['days_after_first_dose'].values.flatten().astype(np.float32)
    return t_measurements


def get_dosing_time_points() -> np.ndarray:
    """sample the dosing time points"""
    df_dosage = pd.read_csv(path_to_data + 'bf_data_dosages_unique.csv')
    dosage = df_dosage.sample(1)[['day_1dose', 'day_2dose', 'day_3dose']].values.flatten().astype(np.float32)

    # convert nan into maximal value
    dosage[np.isnan(dosage)] = NAN_DOSING

    return dosage
