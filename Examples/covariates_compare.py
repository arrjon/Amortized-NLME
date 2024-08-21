#%% md
# # Amortized Inference for a NLME Model

import os
run_i = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))


#%%
import numpy as np
import pandas as pd
from pypesto import Problem, sample
from pypesto.objective import Objective, AggregatedObjective, NegLogParameterPriors
from scipy import stats

from inference.helper_functions import (create_mixed_effect_model_param_names,
                                        analyse_correlation_in_posterior,
                                        create_fixed_params)
from inference.inference_functions import run_population_optimization
#%%
# specify which model to use
model_name = ['fröhlich-simple', 'fröhlich-detailed', 'fröhlich-sde', 
              'pharmacokinetic_model', 
              'dePillis'][-1]
#%% md
# # Load individual model
#%%
prior_type = ['normal', 'uniform'][0]
if model_name == 'fröhlich-simple':
    from models.froehlich_model_simple import FroehlichModelSimple
    individual_model = FroehlichModelSimple(load_best=True, prior_type=prior_type)
elif model_name == 'fröhlich-detailed':
    from models.froehlich_model_detailed import FroehlichModelDetailed
    individual_model = FroehlichModelDetailed(load_best=True, prior_type=prior_type)
elif model_name == 'fröhlich-sde':
    from models.froehlich_model_sde import FroehlichModelSDE
    individual_model = FroehlichModelSDE(load_best=True, prior_type=prior_type)    
elif model_name == 'pharmacokinetic_model':
    from models.pharmacokinetic_model import PharmacokineticModel
    individual_model = PharmacokineticModel(load_best=True)    
elif model_name == 'dePillis':
    from models.de_pillis_model import dePillisModel
    individual_model = dePillisModel(load_best=True)
else:
    raise NotImplementedError('model not implemented')

# assemble simulator and prior
trainer = individual_model.build_trainer('../networks/' + individual_model.network_name)
#%% md
# ## Load Data
#%%
# define how many data points are used for optimization
n_data = 1000
load_real_data = True
# load data
true_pop_parameters = None
results_to_compare = None
if 'Froehlich' in individual_model.name:
    obs_data = individual_model.load_data(n_data=n_data, synthetic=not load_real_data, 
                                          load_egfp=True, load_d2egfp=False)  # if both are loaded, a 2d-list is returned
    if not load_real_data:
        true_pop_parameters = individual_model.load_synthetic_parameter(n_data=n_data)
    
    # load SDE data for comparison
    #from models.froehlich_model_sde import FroehlichModelSDE
    #model_sde = FroehlichModelSDE(load_best=True)
    #obs_data = model_sde.load_data(n_data=n_data, synthetic=True)
    #true_pop_parameters_sde = model_sde.load_synthetic_parameter(n_data=n_data)
else:
    if load_real_data:
        obs_data = individual_model.load_data(n_data=n_data)
    else:
        obs_data, true_pop_parameters = individual_model.load_data(n_data=n_data, synthetic=True, 
                                                                   return_synthetic_params=True)

n_data = len(obs_data)  # in case less data is available
print(len(obs_data), 'individuals')
#
cov_type = ['diag', 'cholesky'][0]
use_covariates = True

mixed_effect_params_names = create_mixed_effect_model_param_names(individual_model.param_names, 
                                                                  cov_type=cov_type)
#%%
# build covariate mapping if needed
covariates_bounds = None
covariate_mapping = None
n_covariates_params = 0
covariates = None
covariates_names = []

if use_covariates and 'fröhlich' in model_name:
    # experiment specific gamma
    gamma_index = [ni for ni, name in enumerate(mixed_effect_params_names) if 'gamma' in name]
    gamma_index_cov = [ni for ni, name in enumerate(mixed_effect_params_names[individual_model.n_params:]) if 'gamma' in name]
    covariates_names = [name + '-d2eGFP' for name in mixed_effect_params_names if 'gamma' in name]
    n_covariates_params = len(covariates_names)
    covariates_bounds = np.array([[-5, 5]] * n_covariates_params)
    
    mixed_effect_params_names = mixed_effect_params_names + covariates_names
    
elif use_covariates and 'dePillis' in model_name:
    covariates_names = ['c_Ab0_not_infection_before_dose', 'c_Ab0_not_infection_before_dose_neg_log_std']
    
    # Marginal Likelihood
    # full model without covariates: 12760-12870
    # full model with covariates: 12858
    # reduced model (excluding : 11177.08
    
    # no additional covariates
    # covariates_list_de_pillis = []
    
    # full model
    covariates_list_de_pillis = [
        'c_Ab0_age', 
        'c_Ab0_male',
        'c_r1_prev_infected', 
        'c_r1_age', 
        'c_r1_male',
        'c_r2_prev_infected', 
        'c_r2_age', 
        'c_r2_male',
        'c_r3_prev_infected', 
        'c_r3_age', 
        'c_r3_male',
        'c_r4_prev_infected', 
        'c_r4_age', 
        'c_r4_male', 
        'c_k1_prev_infected', 
        'c_k1_age',
        'c_k1_male',
        'c_k2_prev_infected', 
        'c_k2_age', 
        'c_k2_male'
    ]

    # Get all possible combinations of all lengths
    import itertools
    all_combinations = []
    for r in range(1, 3):
        combinations_r = itertools.combinations(covariates_list_de_pillis, r)
        all_combinations.extend(combinations_r)

    # Convert to a list
    all_combinations = [()] + list(all_combinations)
    covariates_list_de_pillis = all_combinations[run_i]
    print('Covariates:', covariates_list_de_pillis)

    covariates_dict_de_pillis = {name: i+len(covariates_names) for i, name in enumerate(covariates_list_de_pillis)}
    covariate_param_mapping = {name: i for name in covariates_list_de_pillis 
                               for i, p in enumerate(individual_model.param_names) if p in name}
    covariate_data_mapping = {name: i for name in covariates_list_de_pillis 
                               for i, c in enumerate(['prev_infected', 'age', 'male']) if c in name}
    
    covariates_names += list(covariates_dict_de_pillis.keys())
    n_covariates_params = len(covariates_names)
    covariates_bounds = np.array([[-10, 10], [-3, -np.log(0.01)]] + [[-5, 5]]*len(covariates_dict_de_pillis))
    
    mixed_effect_params_names = mixed_effect_params_names + covariates_names
    assert len(covariates_names) == covariates_bounds.shape[0]

#%%
if use_covariates and 'fröhlich' in model_name:
    # obs_data consists of two groups, first group is eGFP, second group is d2eGFP
    if covariates is None:
        assert len(obs_data) == 2, 'you should load two groups of data'
        covariates = np.concatenate((np.zeros(len(obs_data[0])), np.ones(len(obs_data[1]))))[:, np.newaxis]
        obs_data = np.concatenate((obs_data[0], obs_data[1]))
        n_data = len(obs_data)
        
    from inference.nlme_objective import get_inverse_covariance
    def multi_experiment_mapping(beta: np.ndarray,
                                 psi_inverse_vector: np.ndarray,
                                 covariates: np.ndarray,
                                 covariates_params: np.ndarray):
        """individual_param_i = gamma_{eGFP} * (1-c) + gamma_{d2eGFP} * c + random_effect_{eGFP}, c in {0,1}"""        
        # add param_of_cov*covariates to parameter gamma
        # covariate_params[0] > 0 expected since lifetime of d2eGFP is lower than eGFP
        beta_transformed = np.repeat(beta[np.newaxis, :], covariates.shape[0], axis=0)
        psi_inverse_vector_transformed = np.repeat(psi_inverse_vector[np.newaxis, :], covariates.shape[0], axis=0)
        psi_inverse_transformed = np.zeros((covariates.shape[0], beta.shape[0], beta.shape[0]))
                   
        # flatten since only one covariate     
        beta_transformed[:, gamma_index[0]] = beta_transformed[:, gamma_index[0]] * (1-covariates.flatten()) + covariates_params[0] * covariates.flatten()
        for i, c_i in enumerate(gamma_index_cov):
            psi_inverse_vector_transformed[:, c_i] = psi_inverse_vector[c_i] * (1-covariates.flatten()) + covariates_params[1+i] * covariates.flatten()
        
        for s_id in range(covariates.shape[0]):
            psi_inverse = get_inverse_covariance(psi_inverse_vector_transformed[s_id],
                                                 covariance_format=cov_type,
                                                 param_dim=beta.shape[0])
            psi_inverse_transformed[s_id, :, :] = psi_inverse
        return beta_transformed, psi_inverse_transformed
    
    covariate_mapping = multi_experiment_mapping
    
elif use_covariates and 'dePillis' in model_name:
    _, covariates = individual_model.load_data(n_data, synthetic=not load_real_data, 
                                            load_covariates=True)
    if len(covariates_dict_de_pillis) == 0:
        covariates = covariates[:, 0][:, np.newaxis]
    
    from inference.nlme_objective import get_inverse_covariance
    def previous_infection_mapping(beta: np.ndarray,
                                   psi_inverse_vector: np.ndarray,
                                   covariates: np.ndarray,
                                   covariates_params: np.ndarray):
        beta_transformed = np.repeat(beta[np.newaxis, :], covariates.shape[0], axis=0)
        psi_inverse_vector_transformed = np.repeat(psi_inverse_vector[np.newaxis, :], covariates.shape[0], axis=0)
        psi_inverse_transformed = np.zeros((covariates.shape[0], beta.shape[0], beta.shape[0]))
    
        # covariates are infection status, age, gender
        for k, c_i in covariates_dict_de_pillis.items():
            p_index = covariate_param_mapping[k]
            beta_transformed[:, p_index] += covariates_params[c_i] * covariates[:,  covariate_data_mapping[k]]
        
        # Ab0 for not infected before dose
        beta_transformed[covariates[:, 0] == 0, 0] = covariates_params[0]
        psi_inverse_vector_transformed[covariates[:, 0] == 0, 0] = covariates_params[1]
        for s_id in range(covariates.shape[0]):
            psi_inverse = get_inverse_covariance(psi_inverse_vector_transformed[s_id],
                                                 covariance_format=cov_type,
                                                 param_dim=beta.shape[0])
            psi_inverse_transformed[s_id, :, :] = psi_inverse
        return beta_transformed, psi_inverse_transformed
    
    covariate_mapping = previous_infection_mapping
#%% md
# ### Analyse correlations between parameters
#%%
if cov_type == 'cholesky':
    # if covariance matrix is diagonal, no pairs will appear here since they are not in the mixed effects parameters names list
    high_correlation_pairs = analyse_correlation_in_posterior(model=individual_model, 
                                                              mixed_effect_params_names=mixed_effect_params_names, 
                                                              obs_data=obs_data,
                                                              threshold_corr=0.3)
    print('Parameter pairs of high correlation in individual posterior:', np.array(mixed_effect_params_names)[high_correlation_pairs])
else:
    high_correlation_pairs = []
#%% md
# ## Fixed and Random Effects
# 
# Decide which parameters to fix
# - a fixed effect is modeled as a random effect with variance 0 (all parameters follow a normal distribution)
# - variance of error parameters in the individual model are usually supposed to be a fixed parameter in the population model
# - correlations with these error parameters are usually fixed to 0
# 
#%%
if 'Froehlich' in individual_model.name:
    # fix variance of error parameters and correlations with sigma if cholesky covariance is used
    fix_names = ['std-$\sigma$'] + [name for name in mixed_effect_params_names if '\sigma' in name and 'corr_' in name]
    fixed_values = [0] * len(fix_names)
elif 'Pharmacokinetic' in individual_model.name:
    fix_error_std = ['std-$\\theta_{12}$', 'std-$\\theta_{13}$']
    fix_error_std_val = [0] * len(fix_error_std)
    # fix variance of fixed parameters
    fixed_effects_std = ['std-$\\theta_1$', 'std-$\\theta_5$', 'std-$\\theta_7$', 'std-$\\theta_8$', 
                         'std-$\\theta_{10}$', 'std-$\\theta_{12}$', 'std-$\\theta_{13}$']
    fixed_effects_std_val = [0] * len(fixed_effects_std)
    # fix mean of random effect
    random_effect_mean = ['pop-$\eta_4$']
    random_effect_mean_val = [0]
    
    # put all fixed parameters together
    fix_names = fix_error_std + fixed_effects_std + random_effect_mean
    fixed_values = fix_error_std_val + fixed_effects_std_val + random_effect_mean_val
    
    # if correlations are used, only allow the same as in the original model
    # hence correlations with the error parameter are fixed as well
    if cov_type == 'cholesky':
        non_fix_corr = ['corr_$\\theta_2-\\eta_1$_$\\theta_6-\\eta_2$', 
                        'corr_$\\theta_4-\\eta_3$_$\\theta_6-\\eta_2$', 
                        'corr_$\\theta_4-\\eta_3$_$\\eta_4$']
        fixed_corr = [x for x in mixed_effect_params_names if 'corr_' in x and x not in non_fix_corr]
        fix_names += fixed_corr
        fixed_values += [0] * len(fixed_corr)
    
elif 'dePillis' in individual_model.name:
    fix_names = ['pop-k2', 'std-error_prop', 'c_Ab0_not_infection_before_dose', 'c_Ab0_not_infection_before_dose_neg_log_std']
    fixed_values = [np.log(55.), 0, np.log(0.01), -np.log(0.01)]
    if cov_type == 'cholesky':
        # fix correlations with error parameters
        fix_names += [name for name in mixed_effect_params_names if 'corr_' in name and 'error' in name]
        fixed_values += [0] * len([name for name in mixed_effect_params_names if 'corr_' in name and 'error' in name])
else:
    raise NotImplementedError('model not yet implemented')
    
# "fix" is here in the context of parameters which are not optimized
fixed_indices, fixed_values = create_fixed_params(fix_names=fix_names, 
                                                  fixed_values=fixed_values,
                                                  params_list=mixed_effect_params_names, 
                                                  fix_low_correlation=True,  # only applies to cholesky covariance
                                                  high_correlation_pairs=high_correlation_pairs)
#print(mixed_effect_params_names)
# note: inf values in fixed_values will be set to upper or lower bound respectively

#%%
fixed_indices, unique_indices = np.unique(np.array(fixed_indices), return_index=True)
fixed_values = np.array(fixed_values)[unique_indices]
#%% raw
# # fix all covariate parameters to correct values
# fixed_indices = np.append(fixed_indices, np.arange(len(mixed_effect_params_names)-20, len(mixed_effect_params_names)))
# fixed_values = np.append(fixed_values, np.ones(20)*0.5)
#%% md
# # Run Population Optimization
#%%
import tensorflow as tf
np.random.seed(0)
tf.random.set_seed(0)

pypesto_result, obj_fun_amortized, pesto_problem = run_population_optimization(
    individual_model=individual_model,
    data=obs_data,
    param_names=mixed_effect_params_names,
    cov_type=cov_type,
    n_multi_starts=10,
    n_samples_opt=100,
    covariates_bounds=covariates_bounds,
    covariates=covariates,
    n_covariates_params=n_covariates_params,
    covariate_mapping=covariate_mapping,
    huber_loss=True,
    x_fixed_indices=fixed_indices,
    x_fixed_vals=fixed_values,
    file_name=f'output/{model_name}_real_data_covariates_{run_i}.hdf5',
    verbose=True,
    trace_record=False,
    pesto_multi_processes=1,
    use_result_as_start=False,
    #result=pypesto_result,
)

print(pypesto_result.optimize_result.summary())
BIC = len(pypesto_result.problem.x_free_indices) * np.log(n_data) + 2 * pypesto_result.optimize_result.fval[0]
print('BIC', BIC)
#
# # Bayesian Sampling
# 
# Since our amortized inference is very efficient, we can use it to sample from the posterior distribution of the population parameters.
#%%
# build neg log prior
prior = NegLogParameterPriors(  # no prior is uniform prior on parameter scale defined by bounds
    [
        # population mean gets same prior as individual model
        {'index': pesto_problem.full_index_to_free_index(i),
         'density_fun': lambda x: -stats.norm.logpdf(x, loc=individual_model.prior_mean[i],
                                                     scale=np.sqrt(individual_model.prior_cov.diagonal()[i]))}
        for i in range(individual_model.n_params) if i not in fixed_indices
    ]
    # + [ # std get a half-t prior (non-informative prior)
    #    {'index': pesto_problem.full_index_to_free_index(i),
    #     'density_fun': lambda x: -stats.t.pdf(np.exp(-x), 3, loc=0, scale=1) * 2}
    #    for i in range(individual_model.n_params, individual_model.n_params*2) if i not in fixed_indices
    # ]
    + [  # variance gets a gamma prior (for smaller variances)
        {'index': pesto_problem.full_index_to_free_index(i),
         'density_fun': lambda x: -stats.gamma.pdf(np.exp(-2*x), a=2, scale=1)}
        for i in range(individual_model.n_params, individual_model.n_params * 2) if i not in fixed_indices
    ]
    + [  # all other parameters get a uniform prior
        {'index': pesto_problem.full_index_to_free_index(i),
         'density_fun': lambda x: np.log(pesto_problem.ub_full[i] - pesto_problem.lb_full[i])}
        for i in range(individual_model.n_params * 2, len(pypesto_result.problem.x_names)) if i not in fixed_indices
    ]

    # + [ # negative log of population std gets a normal prior
    #  {'index': pesto_problem.full_index_to_free_index(i+individual_model.n_params), 'density_fun': lambda x: -stats.norm.logpdf(x, loc=inv_std_prior_mean[i], scale=inv_std_prior_std[i])}
    # for i in range(individual_model.n_params) if i+individual_model.n_params not in fixed_indices
    # ] + [  # all other parameters get a standard normal prior
    #    {'index': pesto_problem.full_index_to_free_index(i+individual_model.n_params*2), 'density_fun': lambda x: -stats.norm#.logpdf(x, loc=0, scale=1)}
    # for i in range(len(mixed_effect_params_names)-individual_model.n_params*2) if i+individual_model.n_params*2 not in fixed_indices
    # ]
)
#%%
bayesian_problem = Problem(
    objective=AggregatedObjective([Objective(obj_fun_amortized), prior]),
    lb=pesto_problem.lb_full, 
    ub=pesto_problem.ub_full, 
    x_names=pesto_problem.x_names,
    x_scales=pesto_problem.x_scales,
    x_fixed_indices=pesto_problem.x_fixed_indices,
    x_fixed_vals=pesto_problem.x_fixed_vals,
    x_priors_defs=prior,
)
#%%

i = 0
while True:
    print('Round', i)
    n_samples = 10000
    sampler = sample.ParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        n_chains=30
    )

    pypesto_sampling_result = sample.sample(
        bayesian_problem,
        n_samples=n_samples,
        sampler=sampler,
        x0=pesto_problem.get_reduced_vector(pypesto_result.optimize_result.x[0]),
        filename=f'output/sampling_{model_name}_real_data_covariates_{run_i}.hdf5',
        overwrite=True
    )
    burn_in = sample.geweke_test(pypesto_sampling_result)

    if burn_in < n_samples + 1:
        break
    i += 1
#%%
try:
    harmonic_mean_log_evidence = sample.estimate_evidence.harmonic_mean_log_evidence(pypesto_sampling_result)
    print('Harmonic Mean Log Evidence:', harmonic_mean_log_evidence)

    try:
        bridge_marginal_likelihood = sample.estimate_evidence.bridge_sampling(pypesto_sampling_result,
                                                                              initial_guess_log_evidence=harmonic_mean_log_evidence)
        print('Bridge Marginal Likelihood:', bridge_marginal_likelihood)
    except:
        bridge_marginal_likelihood = np.nan
except:
    harmonic_mean_log_evidence = np.nan
    bridge_marginal_likelihood = np.nan
try:
    ti_marginal_likelihood = sampler.compute_log_evidence(pypesto_sampling_result, use_all_chains=False)
    print('TI Marginal Likelihood:', ti_marginal_likelihood)
except:
    ti_marginal_likelihood = np.nan

# save marginal likelihoods
marginal_likelihoods_df = pd.read_csv('output/marginal_likelihoods.csv')
marginal_likelihoods_df.loc[marginal_likelihoods_df['model'] == run_i, 'BIC'] = BIC
marginal_likelihoods_df.loc[marginal_likelihoods_df['model'] == run_i, 'harmonic mean'] = harmonic_mean_log_evidence
marginal_likelihoods_df.loc[marginal_likelihoods_df['model'] == run_i, 'bridge sampling'] = bridge_marginal_likelihood
marginal_likelihoods_df.loc[marginal_likelihoods_df['model'] == run_i, 'thermodynamic integration'] = ti_marginal_likelihood
marginal_likelihoods_df.to_csv('output/marginal_likelihoods.csv', index=False)
print('Done!')
