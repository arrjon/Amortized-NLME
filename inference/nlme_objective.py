#!/usr/bin/env python
# coding: utf-8

from inspect import signature
from typing import Optional
import numpy as np
from numba import njit, prange
from scipy.linalg import ldl as ldl_decomposition
from scipy.special import logsumexp, erf, gamma


@njit(parallel=True)
def compute_log_integrand_njit(n_sim: int,
                               n_samples: int,
                               log_param_samples: np.ndarray,
                               beta: np.ndarray,  # beta is the mean of the population distribution
                               psi_inverse: np.ndarray,  # psi_inverse is the inverse covariance of the population
                               prior_mean: np.ndarray = None,  # (only really needed for gaussian prior)
                               prior_cov_inverse: Optional[np.ndarray] = None,  # None if uniform prior
                               huber_loss_delta: Optional[float] = None
                               # 1.5 times the median of the standard deviations
                               ) -> np.ndarray:
    """
        compute log of integrand for Monte Carlo approximation of the expectation
        function is vectorized over simulations and samples ( using numba)
    """
    expectation_approx = np.zeros((n_sim, n_samples))

    # for the huber loss we need the cholesky decomposition of psi_inverse
    # cholesky_psi = None  # is not understood by numba
    if huber_loss_delta is not None and psi_inverse.ndim == 2:
        cholesky_psi = np.linalg.cholesky(psi_inverse)

    for sim_id in prange(n_sim):

        if huber_loss_delta is not None and psi_inverse.ndim == 3:
            # psi_inverse might change for every data point
            cholesky_psi = np.linalg.cholesky(psi_inverse[sim_id])

        for sample_id in prange(n_samples):
            if beta.ndim == 1:
                dif = log_param_samples[sim_id, sample_id] - beta
            else:
                # beta might change for every data point
                dif = log_param_samples[sim_id, sample_id] - beta[sim_id]

            # the prior mean does not change for every data point
            dif_prior = log_param_samples[sim_id, sample_id] - prior_mean

            if huber_loss_delta is None:
                # compute quadratic loss
                # psi_inverse can be either the inverse covariance or the transformed inverse covariance
                if psi_inverse.ndim == 2:
                    # quadratic loss for every data point individually
                    temp_psi = 0.5 * dif.T.dot(psi_inverse).dot(dif)
                else:
                    # if psi_inverse is the transformed inverse covariance,
                    # it changes for every data point
                    temp_psi = 0.5 * dif.T.dot(psi_inverse[sim_id]).dot(dif)
            else:
                # compute huber loss
                dif_psi_norm = np.linalg.norm(dif.T.dot(cholesky_psi))
                if np.abs(dif_psi_norm) <= huber_loss_delta:
                    temp_psi = 0.5 * (dif_psi_norm ** 2)
                else:
                    temp_psi = huber_loss_delta * np.abs(dif_psi_norm) - 0.5 * (huber_loss_delta ** 2)

            if prior_cov_inverse is not None:  # gaussian prior
                temp_sigma = 0.5 * dif_prior.T.dot(prior_cov_inverse).dot(dif_prior)
                expectation_approx[sim_id, sample_id] = temp_sigma - temp_psi
            else:
                # uniform prior
                expectation_approx[sim_id, sample_id] = -temp_psi
    return expectation_approx


def compute_log_integrand(n_sim: int,
                          n_samples: int,
                          log_param_samples: np.ndarray,
                          beta: np.ndarray,  # beta is the mean of the population distribution
                          psi_inverse: np.ndarray,  # psi_inverse is the inverse covariance of the population
                          prior_mean: np.ndarray = None,  # (only really needed for gaussian prior)
                          prior_cov_inverse: Optional[np.ndarray] = None,  # None if uniform prior
                          huber_loss_delta: Optional[float] = None
                          # 1.5 times the median of the standard deviations
                          ) -> np.ndarray:
    """
        compute log of integrand for Monte Carlo approximation of the expectation
        function is vectorized over simulations and samples (faster than using numba for a single simulation)
    """
    log_param_samples_reshaped = log_param_samples.reshape(n_sim * n_samples, log_param_samples.shape[2])

    # beta can be only the means or the transformed means containing covariates
    # if beta is the transformed mean, it changes for every data point (and is repeated for every sample)
    dif = log_param_samples_reshaped - beta
    dif_prior = log_param_samples_reshaped - prior_mean

    if huber_loss_delta is None:
        # compute quadratic loss
        # psi_inverse can be either the inverse covariance or the transformed inverse covariance
        if psi_inverse.ndim == 2:
            # quadratic loss for every data point individually, temp_psi is a vector of length n_sim * n_samples
            temp_psi = 0.5 * (dif.dot(psi_inverse) * dif).sum(axis=1)
        else:
            # if psi_inverse is the transformed inverse covariance,
            # it changes for every data point
            temp_psi = np.zeros(n_sim * n_samples)
            for p_i in range(n_sim):
                start, end = p_i * n_samples, (p_i + 1) * n_samples
                temp_psi[start:end] = 0.5 * (dif[start:end].dot(psi_inverse[p_i]) * dif[start:end]).sum(axis=1)
    else:
        # compute huber loss  # todo: does not work with covariates yet
        cholesky_psi = np.linalg.cholesky(psi_inverse)
        temp_psi = np.zeros(n_sim * n_samples)
        for p_i, params in enumerate(log_param_samples_reshaped):  # todo: not yet vectorized
            dif_huber = params - beta
            dif_psi_norm = np.linalg.norm(dif_huber.T.dot(cholesky_psi))
            if np.abs(dif_psi_norm) <= huber_loss_delta:
                temp_psi[p_i] = 0.5 * dif_psi_norm ** 2
            else:
                temp_psi[p_i] = huber_loss_delta * (np.abs(dif_psi_norm) - 0.5 * huber_loss_delta)

    if prior_cov_inverse is not None:
        temp_sigma = 0.5 * (dif_prior.dot(prior_cov_inverse) * dif_prior).sum(axis=1)
        expectation_approx = temp_sigma - temp_psi
    else:
        expectation_approx = -temp_psi

    return expectation_approx.reshape(n_sim, n_samples)


# define objective class
class ObjectiveFunctionNLME:
    """
    objective function for non-linear mixed effects models with normal population distribution
    prior can be either gaussian or uniform
    """

    def __init__(self,
                 model_name: str,
                 prior_mean: np.ndarray,  # needed for gaussian prior
                 prior_std: Optional[np.ndarray] = None,  # needed for gaussian prior
                 param_samples: Optional[np.ndarray] = None,  # (n_sim, n_samples, n_posterior_params)
                 covariance_format: str = 'diag',
                 covariates: Optional[np.ndarray] = None,
                 covariate_mapping: Optional[callable] = None,
                 n_covariates_params: int = 0,
                 joint_model_term: Optional[callable] = None,
                 n_joint_params: int = 0,
                 correlation_penalty: Optional[float] = None,
                 huber_loss_delta: Optional[float] = None,
                 prior_type: str = 'normal',
                 prior_bounds: Optional[np.ndarray] = None,
                 use_njit: bool = True):
        """

        :param model_name: name of model
        :param prior_mean: numpy array of prior means
        :param prior_std: numpy array of prior standard deviations (only needed for gaussian prior)
        :param param_samples: numpy array of parameter samples (can be updated later via update_param_samples)
        :param covariance_format: either 'diag' or 'cholesky'
        :param covariates: numpy array of covariates
        :param covariate_mapping: function that maps parameters, covariates to distributional parameters
        :param n_covariates_params: number of parameters for covariates function
        :param joint_model_term: additional term used to construct a joint model
        :param correlation_penalty: l1 penalty for correlations
        :param huber_loss_delta: delta for huber loss (e.g. 1.5 times the median of the standard deviations,
                        penalizes outliers more strongly than a normal distribution)
        :param prior_type: either 'normal' or 'uniform'
        :param prior_bounds: numpy array of uniform prior bounds (only needed for uniform prior)
        :param use_njit: whether to use numba to speed up computation, default is True,
            depending on the available cores and infrastructure, numba might be slower than numpy
        """

        self.model_name = model_name
        self.param_samples = param_samples if param_samples is not None else np.empty((1, 1, 1))
        self.covariance_format = covariance_format
        if covariance_format != 'diag' and covariance_format != 'cholesky':
            raise ValueError(f'covariance_format must be either "diag" or "cholesky", but is {covariance_format}')
        self.correlation_penalty = correlation_penalty
        self.huber_loss_delta = huber_loss_delta

        self.prior_type = prior_type
        self.prior_mean = prior_mean

        # compute prior terms
        if self.prior_type == 'normal':
            self.prior_cov_inverse = np.diag(1. / prior_std ** 2)
            self.prior_bounds = None
        elif self.prior_type == 'uniform':
            # log( (b-a)^n )
            self.prior_cov_inverse = None
            self.prior_bounds = prior_bounds
        else:
            raise ValueError(f'prior_type must be either "gaussian" or "uniform", but is {self.prior_type}')

        # shape of param_samples
        self.n_sim, self.n_samples, n_posterior_params = self.param_samples.shape
        self.log_n_samples = np.log(self.n_samples)
        # number of parameters for the population distribution (does not include covariates)
        self.param_dim = self.prior_mean.size

        # compute constant terms of loss
        self._set_constants()  # depends on sample size, must be updated if sample changes

        # prepare covariates
        self.covariates = covariates
        # maps samples parameters, covariates to distributional parameters
        self.covariate_mapping = covariate_mapping
        if covariates is not None:
            assert covariate_mapping is not None, '"covariate_mapping" must be specified if covariates are given'
            self.n_covariates_params = n_covariates_params
            assert self.n_covariates_params >= covariates.shape[1], \
                'every covariate must have a parameter (can be fixed)'
            args = list(signature(covariate_mapping).parameters.keys())
            assert 'beta' in args, 'to use a covariate mapping the argument "beta" is expected'
            assert 'psi_inverse' in args, 'to use covariate mapping the argument "psi_inverse" is expected'
            assert 'covariates' in args, 'to use a covariate mapping the argument "covariates" is expected'
            assert 'covariates_params' in args, ('to use a covariate mapping the argument "covariates_params" '
                                                 'is expected')
        else:
            self.n_covariates_params = 0
            self.covariate_mapping = None

        # prepare joint model
        self.joint_model_term = joint_model_term
        self.n_joint_params = n_joint_params
        if joint_model_term is not None:
            args = list(signature(joint_model_term).parameters.keys())
            assert 'param_samples' in args, 'to use a joint model the argument "param_samples" is expected'
            assert 'joint_params' in args, 'to use a joint model the argument "joint_params" is expected'
            assert n_joint_params > 0, 'you need to specify the number of joint model parameters'

        # get indices of parameter types (mean, var, correlation etc.) to construct the right parameter vector
        self.beta_index = range(0, self.param_dim)
        if covariance_format == 'diag':
            last_entry = self.param_dim * 2
        else:
            last_entry = self.param_dim * 2 + (self.param_dim * (self.param_dim - 1)) // 2
        self.psi_inv_index = range(self.param_dim, last_entry)
        if self.n_covariates_params > 0:
            self.covariates_index = range(last_entry, last_entry + self.n_covariates_params)
            last_entry = last_entry + self.n_covariates_params
        else:
            self.covariates_index = None
        if self.n_joint_params > 0:
            self.joint_index = range(last_entry, last_entry + self.n_joint_params)
        else:
            self.joint_index = None

        # set function to use numba or numpy
        if use_njit:
            # depending on the available cores and infrastructure, numba might be slower than numpy
            self.compute_log_integrand = compute_log_integrand_njit
        else:
            self.compute_log_integrand = compute_log_integrand

    def _set_constants(self) -> None:
        # compute constant terms of loss
        if self.prior_type == 'normal' and self.huber_loss_delta is None:
            _, logabsdet = np.linalg.slogdet(self.prior_cov_inverse)
            # constant_prior_term = self.n_sim * 0.5 * (self.param_dim * np.log(2 * np.pi) - logabsdet)
            # constant_population_term = -self.n_sim * 0.5 * self.param_dim * np.log(2 * np.pi)
            self.constant_terms = -self.n_sim * 0.5 * logabsdet
        elif self.prior_type == 'normal' and self.huber_loss_delta is not None:
            _, logabsdet = np.linalg.slogdet(self.prior_cov_inverse)
            constant_prior_term = 0.5 * (self.param_dim * np.log(2 * np.pi) - logabsdet)
            nf = huber_normalizing_factor(delta=self.huber_loss_delta, dim=self.param_dim)
            constant_population_term = -np.log(nf)
            self.constant_terms = self.n_sim * (constant_population_term + constant_prior_term)

        elif self.prior_type == 'uniform' and self.huber_loss_delta is None:
            constant_prior_term = np.sum(np.log(np.diff(self.prior_bounds, axis=1)))
            constant_population_term = -0.5 * self.param_dim * np.log(2 * np.pi)
            self.constant_terms = self.n_sim * (constant_population_term + constant_prior_term)
        elif self.prior_type == 'uniform' and self.huber_loss_delta is not None:
            constant_prior_term = np.sum(np.log(np.diff(self.prior_bounds, axis=1)))
            nf = huber_normalizing_factor(delta=self.huber_loss_delta, dim=self.param_dim)
            constant_population_term = -np.log(nf)
            self.constant_terms = self.n_sim * (constant_population_term + constant_prior_term)

        else:
            raise NotImplementedError('only normal and uniform priors are implemented')
        return

    def update_param_samples(self, param_samples: np.ndarray) -> None:
        """update parameter samples, everything else stays the same"""
        self.param_samples = param_samples
        self.n_sim, self.n_samples, n_posterior_params = param_samples.shape
        self.log_n_samples = np.log(self.n_samples)
        assert n_posterior_params == self.prior_mean.size, 'number of posterior parameters does not match prior'

        if self.n_covariates_params > 0:
            assert self.covariates.shape[0] == self.n_sim, \
                'number of covariates does not match number of simulations'

        # compute constant terms of loss
        self._set_constants()
        return

    def _helper_sum_log_expectation(self,
                                    beta: np.ndarray,
                                    psi_inverse: np.ndarray,
                                    beta_transformed: Optional[np.ndarray] = None,
                                    psi_inverse_transformed: Optional[np.ndarray] = None,
                                    joint_model_params: Optional[np.ndarray] = None,
                                    return_log_integrand: bool = False
                                    ) -> np.ndarray:
        """wrapper function to compute log-sum-exp of second term in objective function with numba"""

        if beta_transformed is None and psi_inverse_transformed is None:
            log_integrand = self.compute_log_integrand(
                n_sim=self.n_sim,
                n_samples=self.n_samples,
                log_param_samples=self.param_samples,
                beta=beta,  # beta is the mean of the population distribution
                psi_inverse=psi_inverse,  # psi_inverse is the inverse covariance of the population distribution
                prior_mean=self.prior_mean,  # (only really needed for gaussian prior)
                prior_cov_inverse=self.prior_cov_inverse,  # None if uniform prior
                huber_loss_delta=self.huber_loss_delta  # optional, only needed for huber loss
            )
        elif beta_transformed is not None and psi_inverse_transformed is None:
            log_integrand = self.compute_log_integrand(
                n_sim=self.n_sim,
                n_samples=self.n_samples,
                log_param_samples=self.param_samples,
                beta=beta_transformed,  # now changes per simulation (and repeated for every sample)
                psi_inverse=psi_inverse,
                prior_mean=self.prior_mean,
                prior_cov_inverse=self.prior_cov_inverse,
                huber_loss_delta=self.huber_loss_delta
            )
        else:  # we assume that psi is never None if beta_transformed is not Non
            log_integrand = self.compute_log_integrand(
                n_sim=self.n_sim,
                n_samples=self.n_samples,
                log_param_samples=self.param_samples,
                beta=beta_transformed,  # now changes per simulation (and repeated for every sample)
                psi_inverse=psi_inverse_transformed,  # now changes per simulation (and repeated for every sample)
                prior_mean=self.prior_mean,
                prior_cov_inverse=self.prior_cov_inverse,
                huber_loss_delta=self.huber_loss_delta
            )

        if self.joint_model_term is not None:
            # compute log of joint model
            log_joint = self.joint_model_term(param_samples=self.param_samples, joint_params=joint_model_params)
            log_integrand += log_joint

        if return_log_integrand:
            return log_integrand

        # log-sum-exp, computes the log of the Monte Carlo approximation of the expectation
        # logsumexp is a stable implementation of log(sum(exp(x)))
        log_sum = logsumexp(log_integrand, axis=1)

        # take sum again over cells/individuals, for each approximation of the expectation subtract log(n_samples)
        sum_log_expectation = np.sum(log_sum) - self.n_sim * self.log_n_samples
        return sum_log_expectation

    # define objective function to minimize parameters
    def __call__(self, vector_params: np.ndarray) -> float:
        # build mean, covariance matrix and vector of parameters for covariates / joint model
        (beta, psi_inverse, psi_inverse_vector,
         covariates_params, joint_model_params) = self.get_params(vector_params=vector_params)

        # include covariates
        if self.n_covariates_params > 0:
            # beta transformed is now a mean depending on the covariates, thus changing for every data point
            transformed_params = self.covariate_mapping(
                beta=beta.copy(),
                psi_inverse=psi_inverse.copy(),
                covariates=self.covariates,
                covariate_params=covariates_params
            )
            # if mapping returns a tuple, the first entry is beta, the second psi_inverse
            # if not the mapping only returns beta
            if isinstance(transformed_params, tuple):
                beta_transformed, psi_inverse_transformed = transformed_params
                # compute parts of the loss
                # now we need to compute the determinant of every psi_inverse for every data point
                det_term = 0
                for sim_idx in range(self.n_sim):
                    _, logabsdet = np.linalg.slogdet(psi_inverse_transformed[sim_idx])
                    det_term += 0.5 * logabsdet
            else:
                beta_transformed, psi_inverse_transformed = transformed_params, None
                # compute parts of the loss
                _, logabsdet = np.linalg.slogdet(psi_inverse)
                det_term = self.n_sim * 0.5 * logabsdet

            # beta_transformed is per simulation
            # we need them in the form of simulation x samples
            beta_transformed = np.repeat(beta_transformed, self.n_samples, axis=0)
        else:
            beta_transformed, psi_inverse_transformed = None, None
            # compute parts of the loss
            _, logabsdet = np.linalg.slogdet(psi_inverse)
            # logabsdet = np.sum(vector_params[self.psi_inv_index][:self.param_dim])
            det_term = self.n_sim * 0.5 * logabsdet

        # compute the loss
        expectation_log_sum = self._helper_sum_log_expectation(
            beta=beta,
            psi_inverse=psi_inverse,
            beta_transformed=beta_transformed,
            psi_inverse_transformed=psi_inverse_transformed,
            joint_model_params=joint_model_params
        )

        # compute negative log-likelihood
        nll = -(self.constant_terms + det_term + expectation_log_sum)

        # add l1 penalty for correlations
        if self.correlation_penalty is not None:
            # the first param_dim entries of psi_inverse_vector are the variances
            nll += self.correlation_penalty * np.sum(np.abs(psi_inverse_vector[self.param_dim:]))
        return nll

    def estimate_mc_integration_variance(self, vector_params: np.ndarray) -> (np.ndarray, np.ndarray):
        """estimate variance of Monte Carlo approximation of the expectation"""
        # build mean, covariance matrix and vector of parameters for covariates / joint model
        (beta, psi_inverse, psi_inverse_vector,
         covariates_params, joint_model_params) = self.get_params(vector_params=vector_params)

        # include covariates
        # include covariates
        if self.n_covariates_params > 0:
            # beta transformed is now a mean depending on the covariates, thus changing for every data point
            transformed_params = self.covariate_mapping(
                beta=beta.copy(),
                psi_inverse=psi_inverse.copy(),
                covariates=self.covariates,
                covariate_params=covariates_params
            )
            # if mapping returns a tuple, the first entry is beta, the second psi_inverse
            # if not the mapping only returns beta
            if isinstance(transformed_params, tuple):
                beta_transformed, psi_inverse_transformed = transformed_params
            else:
                beta_transformed, psi_inverse_transformed = transformed_params, None

            # beta_transformed is per simulation
            # we need them in the form of simulation x samples
            beta_transformed = np.repeat(beta_transformed, self.n_samples, axis=0)
        else:
            beta_transformed, psi_inverse_transformed = None, None

        # compute parts of loss
        log_integrand = self._helper_sum_log_expectation(
            beta=beta,
            psi_inverse=psi_inverse,
            beta_transformed=beta_transformed,
            psi_inverse_transformed=psi_inverse_transformed,
            joint_model_params=joint_model_params,
            return_log_integrand=True
        )
        integrand = np.exp(log_integrand)  # sim x samples

        # log-sum-exp, computes the log of the Monte Carlo approximation of the expectation
        # logsumexp is a stable implementation of log(sum(exp(x)))
        log_expectation = logsumexp(log_integrand, axis=1) - self.log_n_samples
        expectation = np.exp(log_expectation)  # expectation per simulation

        # unbiased estimator of variance of Monte Carlo approximation for each simulation
        var = 1 / (self.n_samples - 1) * np.sum((integrand - expectation[:, np.newaxis]) ** 2, axis=1)  # sim x samples
        error_estimate = np.sqrt(var) / np.sqrt(self.n_samples)
        return var, error_estimate, expectation

    def get_params(self, vector_params: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray,
                                                        Optional[np.ndarray], Optional[np.ndarray]):
        # get parameter vectors
        # vector_params = (beta, psi_inverse_vector, covariates_params)
        # covariates_params and joint_params might be missing

        beta = vector_params[self.beta_index]
        psi_inverse_vector = vector_params[self.psi_inv_index]
        psi_inverse = get_inverse_covariance(psi_inverse_vector,
                                             covariance_format=self.covariance_format,
                                             param_dim=self.param_dim)

        if self.covariates_index is not None:
            covariates_params = vector_params[self.covariates_index]
        else:
            covariates_params = None

        if self.joint_index is not None:
            joint_params = vector_params[self.joint_index]
        else:
            joint_params = None

        return beta, psi_inverse, psi_inverse_vector, covariates_params, joint_params

    def get_inverse_covariance_vector(self, psi: np.ndarray) -> np.ndarray:
        if self.covariance_format == 'diag':
            # psi = log of diagonal entries since entries must be positive
            psi_inv_vector = -np.log(psi.diagonal())
        else:
            # 'cholesky'
            # triangular matrix to vector
            psi_inv = np.linalg.inv(psi)
            lu, d, perm = ldl_decomposition(psi_inv)
            psi_inv_lower = lu[perm, :][np.tril_indices(self.param_dim, k=-1)]
            psi_inv_vector = np.concatenate((np.log(d.diagonal()), psi_inv_lower))
        return psi_inv_vector


def get_inverse_covariance(psi_inverse_vector: np.ndarray,
                           covariance_format: str,
                           param_dim: int) -> np.ndarray:
    if covariance_format == 'diag':
        # psi = log of diagonal entries since entries must be positive
        psi_inverse = np.diag(np.exp(psi_inverse_vector))
    else:
        # matrix is 'cholesky'
        # vector to triangular matrix
        psi_inverse_lower = np.zeros((param_dim, param_dim))
        psi_inverse_lower[np.diag_indices(param_dim)] = 1
        psi_inverse_lower[np.tril_indices(param_dim, k=-1)] = psi_inverse_vector[param_dim:]

        psi_inverse_diag = np.diag(np.exp(psi_inverse_vector[:param_dim]))
        psi_inverse = psi_inverse_lower.dot(psi_inverse_diag).dot(psi_inverse_lower.T)
    return psi_inverse


def get_covariance(psi_inverse_vector: np.ndarray,
                   covariance_format: str,
                   param_dim: int) -> np.ndarray:
    inverse_covariance = get_inverse_covariance(psi_inverse_vector,
                                                covariance_format=covariance_format,
                                                param_dim=param_dim)
    psi = np.linalg.inv(inverse_covariance)
    return psi


def a(n: int, delta: float) -> float:
    if n == 0:
        return np.sqrt(np.pi / 2) * erf(delta / np.sqrt(2))
    elif n == 1:
        return 1 - np.exp(delta**2 / 2)
    else:
        return -(delta**(n-1)) * np.exp(-delta**2 / 2) + (n - 1) * a(n - 2, delta)


def b(n: int, delta: float) -> float:
    if n == 0:
        return np.exp(-delta**2) / delta
    else:
        return (delta**(n-1)) * np.exp(-delta**2) + n * b(n - 1, delta)


def sphere_volume(dim: int, r: float = 1.0) -> float:
    return (np.pi**(dim / 2)) / gamma(dim / 2 + 1) * (r**dim)


def huber_normalizing_factor(delta: float, dim: int) -> float:
    """normalizing factor for huber loss"""
    ad = a(dim - 1, delta)
    bd = b(dim - 1, delta)
    sphere_d_minus_1 = sphere_volume(dim - 1)
    return abs(sphere_d_minus_1) * (ad + np.exp(delta ** 2 / 2) * bd)
