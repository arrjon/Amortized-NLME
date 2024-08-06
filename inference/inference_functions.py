#!/usr/bin/env python
# coding: utf-8

from typing import Union, Optional
import numpy as np
import pandas as pd
from pypesto import Result, Objective, Problem, FD, HistoryOptions, optimize, engine
from tqdm import tqdm

from inference.base_nlme_model import NlmeBaseAmortizer
from inference.helper_functions import create_param_names_opt, create_boundaries_from_prior
from inference.nlme_objective import ObjectiveFunctionNLME


def run_population_optimization(
        individual_model: NlmeBaseAmortizer,
        data: Union[np.ndarray, list[np.ndarray]],
        param_names: list,
        cov_type: str = 'diag',
        n_multi_starts: int = 1,
        n_samples_opt: int = 100,
        param_bounds: Optional[np.array] = None,
        covariates: Optional[np.ndarray] = None,
        covariate_mapping: Optional[callable] = None,
        n_covariates_params: int = 0,
        covariates_bounds: Optional[np.ndarray] = None,
        x_fixed_indices: Optional[np.ndarray] = None,
        x_fixed_vals: Optional[np.ndarray] = None,
        huber_loss: Union[bool, float] = False,
        file_name: Optional[str] = None,
        verbose: bool = False,
        trace_record: bool = False,
        pesto_multi_processes: int = 1,
        pesto_optimizer: optimize.Optimizer = optimize.ScipyOptimizer(),
        use_result_as_start: bool = False,
        result: Optional[Result] = None,
        use_njit: bool = True
) -> (Result, ObjectiveFunctionNLME):
    """
    Run optimization for the population parameters.
    :param individual_model: object of class NlmeBaseAmortizer, used to draw samples from the posterior
    :param data: observed data, should be in the same format as the simulator of the individual model
    :param param_names: list of parameter names
    :param cov_type: can be 'diag' or 'cholesky', determines the covariance structure of the population parameters
    :param n_multi_starts: number of starting points for the optimization
    :param n_samples_opt: number of samples used for the optimization
    :param param_bounds: boundaries for the optimization, if not given, they are automatically created from the prior
    :param covariates: covariates used for the population parameters, will be used in the covariate mapping
    :param covariate_mapping: Function that maps the covariates to the population parameters
    :param n_covariates_params: number of parameters that are used for the covariate mapping,
        might be different from the number of covariates
    :param covariates_bounds: boundaries for the covariates, if not given, they are automatically created from the prior
    :param x_fixed_indices: indices of fixed parameters of the objective function
    :param x_fixed_vals: values of fixed parameters of the objective function
    :param huber_loss: if True, use huber loss for the population parameters, huber loss delta is chosen automatically
        (i.e., 1.5 the median of the standard deviation of the posterior samples)
        if float, use huber loss with given delta
    :param file_name: file name for saving the optimization result (should be hdf5 format)
    :param verbose: if True, print some information during the optimization
    :param trace_record: if True, save the trace of the optimization, needs more memory, but might help to debug
    :param pesto_multi_processes: number of processes used for the optimization, should be <= n_multi_starts, since
        parallel processes use the same objective function, i.e. the same samples from the posterior
    :param pesto_optimizer: optimizer used for the optimization, standard is L-BFGS from scipy
    :param use_result_as_start: if True, use the result from a previous optimization as starting point
    :param result: result object from a previous optimization, if given, the optimization is continued
    :param use_njit: whether to use numba to speed up computation, default is True,
        depending on the available cores and infrastructure, numba might be slower than numpy
    :return:  result object from the optimization, objective function used for the optimization
    """
    # set up huber loss if desired
    if isinstance(huber_loss, float):
        print(f"Use huber loss with delta = {huber_loss}.")
        huber_loss_delta = huber_loss
    elif huber_loss:
        # chose delta in a data dependent way
        samples = individual_model.draw_posterior_samples(data, n_samples_opt).reshape(-1, individual_model.n_params)
        huber_loss_delta = np.round((np.median(np.std(samples, axis=0)) * 1.5), 4)
        print(f"Use huber loss with delta = {huber_loss_delta}.")
    else:
        huber_loss_delta = None
    # create objective function
    obj_fun_amortized = ObjectiveFunctionNLME(model_name=individual_model.name,
                                              prior_mean=individual_model.prior_mean,
                                              prior_std=individual_model.prior_std,
                                              covariance_format=cov_type,
                                              covariates=covariates,
                                              covariate_mapping=covariate_mapping,
                                              n_covariates_params=n_covariates_params,
                                              prior_type=individual_model.prior_type,
                                              # for uniform prior
                                              prior_bounds=individual_model.prior_bounds if hasattr(individual_model,
                                                                                                    'prior_bounds') else None,
                                              huber_loss_delta=huber_loss_delta,
                                              use_njit=use_njit
                                              )
    # set up pyPesto
    param_names_opt = create_param_names_opt(dim=individual_model.amortizer.latent_dim,
                                             cov_type=cov_type,
                                             param_names=param_names)
    # create bounds if none are given
    if param_bounds is None:
        # automatically create boundaries
        param_bounds = create_boundaries_from_prior(
            prior_mean=individual_model.prior_mean,
            prior_std=individual_model.prior_std,
            prior_type=individual_model.prior_type,
            prior_bounds=individual_model.prior_bounds if hasattr(individual_model, 'prior_bounds') else None,
            covariates_bounds=covariates_bounds,  # only used if covariates are used
            covariance_format=cov_type)

        if covariates_bounds is not None:
            assert covariates_bounds.shape[1] == 2, \
                "covariates_bounds should be a 2d array with shape (2, n_covariates)"
            assert covariates_bounds.shape[0] == n_covariates_params, \
                "covariates_bounds should be a 2d array with shape (2, n_covariates)"

    # set up fixed parameters to be unique
    x_fixed_indices, unique_indices = np.unique(np.array(x_fixed_indices), return_index=True)
    x_fixed_vals = np.array(x_fixed_vals)[unique_indices]

    # save optimizer trace
    history_options = HistoryOptions(trace_record=trace_record)

    if pesto_multi_processes > 1:
        # use multiprocessing, but start serval times with new objective function
        n_runs = n_multi_starts // pesto_multi_processes
        if n_runs == 0:
            # if more processes then starts are given, all use the same objective function
            n_runs = 1
        elif n_runs * pesto_multi_processes < n_multi_starts:
            # if not all starts are used, add one more run
            n_runs += 1
        n_starts_per_run = pesto_multi_processes
        pesto_engine = engine.MultiProcessEngine(pesto_multi_processes)
    else:
        n_runs = n_multi_starts
        n_starts_per_run = 1
        pesto_engine = engine.SingleCoreEngine()

    if pesto_multi_processes >= n_multi_starts:
        print("Warning: pesto_multi_processes >= n_multi_starts. All starts use the same samples from the posterior. "
              "This is not recommended and you should increase 'n_multi_starts'.")

    if result is not None:
        n_old_runs = len(result.optimize_result.list)
    else:
        n_old_runs = 0

    pesto_problem = None
    for run_idx in tqdm(range(n_old_runs, n_old_runs + n_runs), disable=not verbose, desc='Multi-start optimization'):
        # run optimization for each starting point with different objective functions (due to sampling)
        # if pesto_multi_processes > 1, same objective function is used for all starting points

        # create objective function with samples
        param_samples = individual_model.draw_posterior_samples(data=data, n_samples=n_samples_opt)
        # update objective function with samples
        obj_fun_amortized.update_param_samples(param_samples=param_samples)

        # sanity check of objective function
        test = obj_fun_amortized(np.ones(len(param_names_opt)))
        assert isinstance(test, float), f"Objective function should return a scalar, but returned {test}."

        pesto_objective = FD(obj=Objective(fun=obj_fun_amortized, x_names=param_names_opt))

        # create pypesto problem
        if use_result_as_start and result is not None:
            # use result from previous optimization as starting point
            print("Use result from previous optimization as starting point.")
            x_guesses = result.optimize_result.x
        else:
            x_guesses = None
        pesto_problem = Problem(objective=pesto_objective,
                                lb=param_bounds[0, :],
                                ub=param_bounds[1, :],
                                x_names=param_names_opt,
                                x_scales=['log'] * len(param_names_opt),
                                x_fixed_indices=x_fixed_indices,
                                x_fixed_vals=x_fixed_vals,
                                x_guesses=x_guesses)
        if verbose and run_idx == n_old_runs:
            pesto_problem.print_parameter_summary()
            df_fixed = pd.DataFrame(pesto_problem.x_fixed_vals,
                                    index=np.array(param_names_opt)[pesto_problem.x_fixed_indices],
                                    columns=['fixed value'])
            print(df_fixed)

        # Run optimizations for different starting values
        result = optimize.minimize(
            problem=pesto_problem,
            result=result,
            optimizer=pesto_optimizer,
            ids=[f"batch_{run_idx}_{i}" for i in range(n_starts_per_run)],
            n_starts=n_starts_per_run,
            engine=pesto_engine,
            history_options=history_options,
            progress_bar=False if pesto_multi_processes <= 1 else verbose,
            filename=file_name if file_name is not None else None,
            overwrite=True if file_name is not None else False)

    # make final objective value comparable across samples
    # always use more samples for final evaluation
    # MC integration error reduces with sqrt(1/n_samples)
    param_samples = individual_model.draw_posterior_samples(data=data, n_samples=n_samples_opt * 10)
    # update objective function with samples
    obj_fun_amortized.update_param_samples(param_samples=param_samples)

    result_list = result.optimize_result.list.copy()
    for res in result_list:
        # check if res is NoneType (failed start)
        if res['x'] is None:
            continue
        res['fval'] = obj_fun_amortized(res['x'])
    setattr(result.optimize_result, "list", result_list)
    result.optimize_result.sort()

    # update objective function with fewer samples (for faster evaluation, e.g. profiling)
    param_samples = individual_model.draw_posterior_samples(data=data, n_samples=n_samples_opt)
    obj_fun_amortized.update_param_samples(param_samples=param_samples)
    if pesto_problem is not None:
        pesto_problem.objective = FD(obj=Objective(fun=obj_fun_amortized, x_names=param_names_opt))
    return result, obj_fun_amortized, pesto_problem

