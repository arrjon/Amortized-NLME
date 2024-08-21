#!/usr/bin/env python
# coding: utf-8

from typing import Optional
import numpy as np

from inference.likelihoods import log_likelihood_additive_noise, log_likelihood_multiplicative_noise


class ObjectiveFunctionEmpiricalBayes:
    def __init__(self,
                 data: np.ndarray,
                 pop_mean: np.ndarray,
                 pop_cov: np.ndarray,
                 sigmas: np.ndarray,
                 batch_simulator: Optional[callable] = None,
                 noise_type: str = "multiplicative",
                 ignore_conditional: bool = False,  # returns only a likelihood
                 huber_loss_delta: Optional[float] = None
                 ):
        self.data = data
        self.pop_mean = pop_mean
        self.pop_cov = pop_cov
        self.sigmas = sigmas
        self.batch_simulator = batch_simulator
        self.ignore_conditional = ignore_conditional

        # precompute some values
        self.inv_pop_cov = np.linalg.inv(pop_cov)
        _, logabsdet = np.linalg.slogdet(self.pop_cov)
        self.precompute_mvn = -0.5 * (self.pop_mean.size * np.log(2 * np.pi) + logabsdet)
        self.noise_type = noise_type

        self.huber_loss_delta = huber_loss_delta
        if huber_loss_delta is not None:
            self.cholesky_psi = np.linalg.cholesky(self.inv_pop_cov)

    def __call__(self, individual_params: np.ndarray, sim_data: Optional[np.ndarray] = None) -> float:

        if sim_data is None:
            # simulate data from the simulator if not provided
            sim_data = self.batch_simulator(individual_params)

        if self.ignore_conditional:
            # then only the likelihood is returned
            conditional = 0
        else:
            dif = individual_params - self.pop_mean
            if self.huber_loss_delta is None:
                # compute multivariate normal logpdf
                temp = dif.T.dot(self.inv_pop_cov).dot(dif)
            else:
                dif_psi_norm = np.linalg.norm(dif.T.dot(self.cholesky_psi))
                if np.abs(dif_psi_norm) <= self.huber_loss_delta:
                    temp = 0.5 * (dif_psi_norm ** 2)
                else:
                    temp = self.huber_loss_delta * np.abs(dif_psi_norm) - 0.5 * (self.huber_loss_delta ** 2)
            conditional = self.precompute_mvn - 0.5 * temp

        # compute multivariate normal logpdf
        if self.noise_type == "additive":
            log_likelihood = log_likelihood_additive_noise(measurements=self.data,
                                                           simulations=sim_data,
                                                           sigmas=self.sigmas)
        else:
            log_likelihood = log_likelihood_multiplicative_noise(log_measurements=self.data,
                                                                 log_simulations=sim_data,
                                                                 sigmas=self.sigmas)
        return -(log_likelihood + conditional)
