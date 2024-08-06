#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
from typing import Optional
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pypesto import visualize, Result
from scipy.stats import lognorm, entropy, norm
from scipy.stats import t as t_dist


def visualize_pesto_result(result: Result,
                           use_batch_coloring: bool = True,
                           obj_fun_amortized: Optional = None) -> None:
    # visualize results of optimization
    if use_batch_coloring:
        # results' id is "batch_id_run_id", so int(s.split('_')[1]) gives us the run_id
        c_id = np.array([int(s.split('_')[1]) for s in result.optimize_result.id])
        cm_mapper = cm.ScalarMappable(norm=cm.colors.Normalize(vmin=0, vmax=np.max(c_id) + 1), cmap=cm.hsv)
        color_map = cm_mapper.to_rgba(c_id)
    else:
        color_map = None

    visualize.waterfall(result, colors=color_map)
    visualize.parameters(result, colors=color_map)
    if result.optimize_result.history[0] is not None and result.optimize_result.history[0].options.trace_record:
        visualize.optimizer_history(result, colors=color_map)

    if obj_fun_amortized is not None:
        var, error_estimate, expectation = obj_fun_amortized.estimate_mc_integration_variance(
            result.optimize_result.x[0]
        )

        fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(10, 5), sharey='all')
        ax[0].hist(expectation, color=color_map[0] if color_map is not None else None)
        ax[0].set_title('Estimated Expectation \nper Individual (best parameters)')
        ax[0].set_xlabel('Expectation')
        ax[0].set_ylabel('Number of Individuals')

        ax[1].hist(error_estimate, color=color_map[0] if color_map is not None else None)
        ax[1].set_title('Estimated Error of MC')
        ax[1].set_xlabel('Error')

        expectation_cut = expectation.copy()
        expectation_cut[expectation_cut == 0] = 1e-10
        rel_error = error_estimate / expectation_cut

        ax[2].hist(rel_error, color=color_map[0] if color_map is not None else None)
        ax[2].set_title('Estimated Relative Error')
        ax[2].set_xlabel('Relative Error')
        plt.show()

        # detect whether the error estimate is too large
        if np.median(rel_error) > 0.5:  # todo: rather arbitrary threshold, should be investigated
            print(f'Warning: The median error estimate is large ({np.median(rel_error)}). '
                  f'Consider increasing the number of samples for the Monte Carlo Integration.')

        # print('Max Approx. Rel. Error of the MC Integration', rel_error.max())
        # using the maximum error estimate as an approximation for the overall error
        # we take the product over the expectations and the then log of it (this we want to maximize)
        # the error of the product is the maximum of the error estimates
    return


def plot_real_and_estimated(estimated_mean: np.ndarray,
                            estimated_cov: np.ndarray,
                            simulator: callable, model_name: str,
                            data: np.ndarray = None,
                            n_trajectories: int = 50,
                            exp_func: callable = np.exp,
                            save_fig: str = None,
                            seed: int = 0) -> None:
    """
    Plots real and estimated trajectories in two subplots next to each other.
    """
    np.random.seed(seed)
    # sample from log normal distribution
    reproduced_param = np.random.multivariate_normal(estimated_mean,
                                                     estimated_cov,
                                                     size=n_trajectories)

    # check if model is Froehlich model
    if 'FroehlichModel' in model_name:
        if data is None:
            raise ValueError('Data must be provided for Froehlich model')
        # plot real synthetic data vs estimated data
        t_points = np.linspace(start=1 / 6, stop=30, num=data.shape[1], endpoint=True)
        reproduced_data = simulator(reproduced_param)
        if isinstance(reproduced_data, dict):
            reproduced_data = reproduced_data['sim_data']

        # Plot
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey='all', sharex='all', figsize=(15, 5))  # , dpi=600)

        for i in range(data.shape[0]):
            ax[0].plot(t_points, exp_func(data[i, :]), color='grey', alpha=0.3)
        for i in range(reproduced_param.shape[0]):
            ax[1].plot(t_points, exp_func(reproduced_data[i]), color='red', alpha=0.3)

        ax[0].set_xlabel('$t\,[h]$')
        ax[1].set_xlabel('$t\,[h]$')
        ax[0].set_ylabel('fluorescence intensity [a.u.]')
        plt.yscale('log')
        ax[0].set_title(f'Single Cell Expression - Data')
        ax[1].set_title(f'Single Cell Expression - Estimated')
        fig.tight_layout()
        if save_fig is not None:
            plt.savefig(f'plots/{save_fig}_' + str(datetime.now()) + '.png')
        plt.show()
    elif model_name == 'PharmacokineticModel':

        # t_measurement = [222.5833333, 387.1666667, 555.5, 699.2833333, 890.75, 1060.016667, 1203.35]
        t_doses = [26, 50, 74, 98, 122, 146, 170, 194, 218,
                   242, 266, 290, 314, 338, 362, 386, 410, 434,
                   458, 482, 506, 530, 554, 578, 602, 626, 650, 674]
        WT = 30
        DOS = 75

        plt.figure(figsize=(10, 10))  # , dpi=600)
        for s_idx, s in enumerate(reproduced_param):
            input_param = np.concatenate((s[:14], np.log([WT, DOS]), s[14:]))
            t_sim, y_sim = simulator(input_param,
                                     t_measurement=np.linspace(0, 1200, 1000),
                                     t_doses=t_doses,
                                     return_all_states=False,
                                     with_noise=True)
            # for i in range(2):
            if s_idx == 0:
                plt.plot(t_sim, np.exp(y_sim[:, 0]), color='red', alpha=0.2, label=f'simulated $A_{2}$')
                plt.plot(t_sim, np.exp(y_sim[:, 1]), color='green', alpha=0.2, label=f'simulated $A_{3}$')
            else:
                plt.plot(t_sim, np.exp(y_sim[:, 0]), color='red', alpha=0.2)
                plt.plot(t_sim, np.exp(y_sim[:, 1]), color='green', alpha=0.2)
        plt.vlines(t_doses, 1, np.exp(1), color='grey')
        plt.yscale('log')
        plt.legend()
        plt.show()
    else:
        raise NotImplementedError('Model not supported')
    return


def plot_real_vs_synthetic(estimated_mean: np.ndarray,
                           estimated_cov: np.ndarray,
                           simulator: callable,
                           model_name: str,
                           data: np.ndarray = None,
                           n_trajectories: int = 50,
                           estimation_function: callable = np.mean,
                           ylim: tuple[float, float] = None,
                           fig_name: Optional[str] = None,
                           seed: int = 0) -> None:
    np.random.seed(seed)
    # sample from normal distribution
    reproduced_param = np.random.multivariate_normal(estimated_mean,
                                                     estimated_cov,
                                                     size=n_trajectories)

    # check if model is Froehlich model
    if 'FroehlichModel' in model_name:
        if data is None:
            raise ValueError('Data must be provided for Froehlich model')
        # plot real synthetic data vs estimated data
        t_points = np.linspace(start=1 / 6, stop=30, num=data.shape[1], endpoint=True)
        reproduced_data = simulator(reproduced_param)
        if isinstance(reproduced_data, dict):
            reproduced_data = reproduced_data['sim_data']

        dif = estimation_function(reproduced_data, axis=0) - estimation_function(data, axis=0)
        std = np.std(reproduced_data, axis=0)
        t_value_05 = abs(t_dist.ppf(0.05 / reproduced_data.shape[1], df=reproduced_data.shape[0] - 1))

        confidence_band_upper_05 = dif + t_value_05 * std / np.sqrt(reproduced_data.shape[0])
        confidence_band_lower_05 = dif - t_value_05 * std / np.sqrt(reproduced_data.shape[0])

        # Plot
        plt.figure()
        plt.fill_between(t_points, confidence_band_upper_05.flatten(), confidence_band_lower_05.flatten(),
                         color='grey', alpha=0.3, label='95% confidence band')
        plt.plot(t_points, dif, color='red', label='estimation difference')
        plt.plot(t_points, np.zeros(t_points.size), color='black', linestyle='--')
        plt.xlabel('$t\,[h]$')
        plt.title(f'Mean Difference')
        plt.legend()
        if ylim is not None:
            plt.ylim(ylim)
        plt.tight_layout()
        if fig_name is not None:
            plt.savefig(fig_name)
        plt.show()
    else:
        raise NotImplementedError('Model not supported')
    return


def plot_parameter_estimates(result_list: list[np.ndarray],
                             param_names_plot: list[str],
                             prior_mean: np.ndarray = None,
                             prior_std: np.ndarray = None,
                             true_parameters: np.ndarray = None,
                             run_names: list[str] = None,  # None, if multi-starts are compared
                             fig_name: Optional[str] = None) -> None:
    # plot parameters
    fig, ax = plt.subplots(figsize=(15, 5))  # , dpi=600)
    parameters_ind = list(range(1, result_list[0].shape[0] + 1))[::-1]

    if prior_mean is not None and prior_std is not None:
        n_pop_params = len(prior_mean)
        prior_interval = np.array([prior_mean - 1.96 * prior_std, prior_mean + 1.96 * prior_std]).T
        ax.fill_betweenx(parameters_ind[:n_pop_params], prior_interval[:, 0], prior_interval[:, 1],
                         color='grey', alpha=0.2, label='95% prior region')

    for j_x, x in reversed(list(enumerate(result_list))):
        if run_names is None:
            if j_x == 0:
                tmp_legend = 'optimal run'
            else:
                tmp_legend = None
        else:
            tmp_legend = run_names[j_x]
        ax.plot(
            x,
            parameters_ind,
            linestyle='dashed',
            # color=colors[j_x],
            marker='o',
            label=tmp_legend,
        )

    if true_parameters is not None:
        ax.plot(true_parameters, parameters_ind, color='red', marker='x',
                label='true parameters sample')

    ax.set_yticks(parameters_ind, param_names_plot)
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Parameter')
    ax.set_title('Estimated Population Parameters (log-normal distribution)')
    ax.legend(loc=2, bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name)
    plt.show()
    return


def plot_estimated_distributions(result_list: list[np.ndarray],
                                 param_names_plot: list[str],
                                 prior_mean: np.ndarray,
                                 prior_std: np.ndarray,
                                 true_parameters: np.ndarray = None,
                                 prior_width: float = 1.98,
                                 fig_name: Optional[str] = None) -> None:
    # plot posteriors for each parameter individually
    n_params_plot = len(param_names_plot)
    fig, ax = plt.subplots(nrows=1, ncols=n_params_plot, figsize=(15, 5))  # , dpi=600)

    for p_idx, name in enumerate(param_names_plot):

        x = np.linspace(np.exp(prior_mean[p_idx] - prior_width * prior_std[p_idx]),
                        np.exp(prior_mean[p_idx] + prior_width * prior_std[p_idx]), 100000)

        # Set parameters for the log-normal distribution
        mu = result_list[p_idx]
        sigma = result_list[n_params_plot + p_idx]

        # Create a log-normal distribution with the given parameters
        log_norm = lognorm(s=sigma, scale=np.exp(mu))

        # Set parameters for the log-normal distribution
        if true_parameters is not None:
            mu_real = true_parameters[p_idx]
            sigma_real = true_parameters[n_params_plot + p_idx]

            # Create true log-normal distribution with the given parameters
            log_norm_real = lognorm(s=sigma_real, scale=np.exp(mu_real))

            # Plot real and estimated density functions
            kl_divergence = entropy(log_norm.pdf(x), log_norm_real.pdf(x))
            ax[p_idx].plot(x, log_norm.pdf(x), 'r', label='estimated')
            ax[p_idx].plot(x, log_norm_real.pdf(x), 'b-.', label='real')
            ax[p_idx].set_title(name + f'\nKL-Div: {round(kl_divergence, 5)}')
        else:
            # Plot estimated density function
            ax[p_idx].plot(x, log_norm.pdf(x), 'r', label='estimated')
            ax[p_idx].set_title(name)

    ax[-2].set_title(param_names_plot[-2])
    ax[-1].set_title(param_names_plot[-1])
    if true_parameters is not None:
        ax[-2].axvline(np.exp(true_parameters[-n_params_plot - 2]), color='b', linestyle='-.')
        ax[-1].axvline(np.exp(true_parameters[-n_params_plot - 1]), color='b', linestyle='-.')
        if n_params_plot == 11:
            ax[3].axvline(np.exp(true_parameters[3]), color='b', linestyle='-.')
            ax[3].set_title(param_names_plot[3])
        elif n_params_plot == 8:
            ax[4].axvline(np.exp(true_parameters[4]), color='b', linestyle='-.')
            ax[4].set_title(param_names_plot[4])

    plt.legend()
    fig.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name)
    plt.show()
    return


def plot_normal_distributions(beta: np.ndarray,
                              psi: np.ndarray,
                              title: Optional[str] = None,
                              label: Optional[str] = None,
                              param_names_plot: Optional[list[str]] = None,
                              posterior_samples: Optional[np.ndarray] = None,
                              lb: Optional[np.ndarray] = None,
                              ub: Optional[np.ndarray] = None,
                              max_col: int = 5,
                              ax: Optional[plt.Axes] = None,
                              color: str = 'blue') -> plt.Axes:
    # plot distributions
    n_cols = min(max_col, beta.size)
    n_rows = int(np.ceil(beta.size / n_cols))
    if ax is None:
        _, ax = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(15, 5))
    axis = ax.flatten()

    # set x limits
    if lb is None:
        lb = beta - 3 * np.sqrt(psi.diagonal())
    else:
        assert lb.size >= beta.size, 'lb must have at least the same size as beta'
    if ub is None:
        ub = beta + 3 * np.sqrt(psi.diagonal())
    else:
        assert ub.size >= beta.size, 'ub must have at least the same size as beta'

    for p_id in range(beta.size):
        # plot normal distribution
        x = np.linspace(beta[p_id] - 2.58 * np.sqrt(psi.diagonal()[p_id]),
                        beta[p_id] + 2.58 * np.sqrt(psi.diagonal()[p_id]), 100)
        axis[p_id].plot(x, norm.pdf(x, loc=beta[p_id], scale=np.sqrt(psi.diagonal()[p_id])),
                        color=color, label=label)

        if posterior_samples is not None:
            axis[p_id].hist(posterior_samples[:, :, p_id].flatten(), bins=20, density=True,
                            label='Posterior Samples')

        axis[p_id].set_xlim((lb[p_id], ub[p_id]))
        if param_names_plot is not None:
            axis[p_id].set_xlabel(param_names_plot[p_id])

    if title is not None:
        axis[n_cols // 2].set_title(title)

    #for _ax in axis[beta.size:]:
    #    _ax.remove()
    return ax


def plot_histograms(param_samples: np.ndarray,
                    title: Optional[str] = None,
                    param_names_plot: Optional[list[str]] = None,
                    lb: Optional[np.ndarray] = None,
                    ub: Optional[np.ndarray] = None,
                    max_col: int = 5) -> None:
    n_params = param_samples.shape[2]
    n_cols = min(max_col, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, tight_layout=True, figsize=(15, 5))
    axis = ax.flatten()

    for p_id in range(n_params):
        axis[p_id].hist(param_samples[:, :, p_id].flatten(), bins=20, density=True)
        axis[p_id].set_xlabel(param_names_plot[p_id])

        if lb is not None and ub is not None:
            axis[p_id].set_xlim((lb[p_id], ub[p_id]))

    if title is not None:
        axis[n_cols // 2].set_title(title)

    for _ax in axis[n_params:]:
        _ax.remove()
    plt.show()
    return


def plot_distribution(result_list: list,
                      param_name_plot: str,
                      result_names: list[str],
                      min_x: float,
                      max_x: float,
                      fig_name: Optional[str] = None) -> None:
    # plot distributions for one parameter
    fig = plt.figure(figsize=(15, 5))  # , dpi=600)

    x = np.linspace(min_x, max_x, 10000)

    for r_idx, result in enumerate(result_list):
        # Set parameters for the log-normal distribution
        mu, sigma = result

        # Create a log-normal distribution with the given parameters
        log_norm = lognorm(s=sigma, scale=np.exp(mu))

        # Plot estimated density function
        plt.plot(x, log_norm.pdf(x), label=result_names[r_idx])
        plt.title(param_name_plot)

    plt.legend()
    plt.xscale('log')
    fig.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name)
    plt.show()
    return


def corrplot(corr_df: pd.DataFrame, ax: Optional[plt.Axes] = None, square: bool = True) -> plt.Axes:
    if ax is None:
        f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_df,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                square=square, ax=ax)
    ax.set_title('Correlation Matrix', fontsize=16)
    return ax
