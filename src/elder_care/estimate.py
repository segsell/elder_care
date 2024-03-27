import pickle


from dcegm.solve import get_solve_func_for_model
from model_code.budget_equation import create_savings_grid
from model_code.specify_model import specify_model

import estimagic as em


def estimate_model():

    # estimagic taking criterion function and parameters

    optm = em.minimize(
        criterion=criterion_solve_and_simulate,
        params=start,
        # params=params_test,
        lower_bounds=lower,
        upper_bounds=upper,
        algorithm="tranquilo_ls",
        algo_options=algo_options,
        multistart=True,
        multistart_options=options,
        constraints={
            "selector": lambda params: {key: params[key] for key in FIXED_PARAMS},
            "type": "fixed",
        },
    )

    # Save result object and parameters
    # result = em.minimize(
    #     criterion=individual_likelihood_print,
    #     params=start_params,
    #     lower_bounds=lower_bounds,
    #     upper_bounds=upper_bounds,
    #     algorithm="scipy_lbfgsb",
    #     logging="test_log.db",
    #     error_handling="continue",
    # )
    # pickle.dump(result, open(path_dict["est_results"] + "em_result_1.pkl", "wb"))
    # start_params_all.update(result.params)
    # pickle.dump(
    #     start_params_all, open(path_dict["est_results"] + "est_params_1.pkl", "wb")
    # )


def setup_estimate_model():

    # 1) Load model

    # 2) Get solve function
    solve_func = get_solve_func_for_model(
        options=options,
        exog_savings_grid=exog_savings_grid,
        utility_functions=utility_functions,
        final_period_functions=utility_functions_final_period,
        budget_constraint=budget_constraint,
        state_space_functions=state_space_functions,
    )

    # 3) Initial conditions
    initial_resources, initial_states = draw_initial_states(
        initial_conditions, initial_wealth_empirical, n_agents, seed=seed
    )

    # 4) Load empirical moments


def criterion_solve_and_simulate(params, n_agents=10_000):

    # ! random seed !
    seed = int(time.time())

    # params = revert_parameters(params, original_min=-5, original_max=5, target_min=-1, target_max=1)

    value, policy_left, policy_right, endog_grid = solve_func(params)

    initial_resources, intial_states = draw_initial_states(
        initial_conditions, initial_wealth_empirical, n_agents, seed=seed - 1
    )

    result = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
        n_periods=options["state_space"]["n_periods"],
        params=params,
        #
        state_space_names=state_space_names,
        seed=seed,
        #
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_left_solved=policy_left,
        policy_right_solved=policy_right,
        #
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        choice_range=jnp.arange(map_state_choice_to_index.shape[-1], dtype=jnp.int16),
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        compute_utility=model_funcs["compute_utility"],
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )

    arr = create_simulation_array(
        result, options=options, params=params, n_agents=n_agents
    )
    _sim_moments = simulate_moments(arr, idx)
    sim_moments = jnp.where(jnp.isnan(_sim_moments), 0, _sim_moments)
    sim_moments = jnp.where(jnp.isinf(sim_moments), 0, sim_moments)

    err = sim_moments - emp_moments
    # crit_val = jnp.dot(jnp.dot(err.T, chol_weights), err)

    # deviations = sim_moments - np.array(emp_moments)
    root_contribs = err @ chol_weights
    crit_val = root_contribs @ root_contribs

    return {"root_contributions": root_contribs, "value": crit_val}


# ==============================================================================


def setup():

    if load_model:
        model = load_and_setup_model(
            options=options,
            state_space_functions=create_state_space_functions(),
            utility_functions=create_utility_functions(),
            utility_functions_final_period=create_final_period_utility_functions(),
            budget_constraint=budget_constraint,
            path=path_dict["intermediate_data"] + "model.pkl",
        )

    model, options, params = specify_model(
        path_dict=path_dict,
        params=start_params_all,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        load_model=load_model,
    )

    # Create savings grid
    savings_grid = create_savings_grid()

    # ==============================================================================

    lower_bounds = {
        # "mu": 1e-12,
        "dis_util_work": 1e-12,
        "dis_util_unemployed": 1e-12,
        # "bequest_scale": 1e-12,
        # "lambda": 1e-12,
    }
    upper_bounds = {
        # "mu": 5,
        "dis_util_work": 50,
        "dis_util_unemployed": 50,
        # "bequest_scale": 20,
        # "lambda": 1,
    }

    result = em.minimize(
        criterion=individual_likelihood_print,
        params=start_params,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        algorithm="scipy_lbfgsb",
        logging="test_log.db",
        error_handling="continue",
    )
    pickle.dump(result, open(path_dict["est_results"] + "em_result_1.pkl", "wb"))
    start_params_all.update(result.params)
    pickle.dump(
        start_params_all, open(path_dict["est_results"] + "est_params_1.pkl", "wb")
    )


def specify_and_solve_model(
    path_dict,
    file_append,
    params,
    update_spec_for_policy_state,
    policy_state_trans_func,
    load_model,
    load_solution,
):
    """Solve the model and save the solution as well as specifications to a file."""
    solution_file = path_dict["intermediate_data"] + (
        f"solved_models/model_solution" f"_{file_append}.pkl"
    )

    # Generate model_specs
    model, options, params = specify_model(
        path_dict=path_dict,
        update_spec_for_policy_state=update_spec_for_policy_state,
        policy_state_trans_func=policy_state_trans_func,
        params=params,
        load_model=load_model,
    )

    if load_solution:
        solution_est = pickle.load(open(solution_file, "rb"))
        return solution_est, model, options, params

    savings_grid = create_savings_grid()

    solve_func = get_solve_func_for_model(model, savings_grid, options)
    value, policy_left, policy_right, endog_grid = solve_func(params)

    solution = {
        "value": value,
        "policy_left": policy_left,
        "policy_right": policy_right,
        "endog_grid": endog_grid,
    }

    pickle.dump(solution, open(solution_file, "wb"))

    return solution, model, options, params


MULTISTART_OPTONS = {
    # Set the number of points at which criterion is evaluated
    # in the exploration phase
    "n_samples": 5 * len(params_test),
    # Pass in a DataFrame or array with a custom sample
    # for the exploration phase.
    "sample": None,
    # Determine number of optimizations, relative to n_samples
    "share_optimizations": 0.1,
    # Determine distribution from which sample is drawn
    "sampling_distribution": "uniform",
    # Determine sampling method. Allowed: ["sobol", "random",
    # "halton", "hammersley", "korobov", "latin_hypercube"]
    "sampling_method": "sobol",
    # Determine how start parameters for local optimizations are
    # calculated. Allowed: ["tiktak", "linear"] or a custom
    # function with arguments iteration, n_iterations, min_weight,
    # and max_weight
    "mixing_weight_method": "tiktak",
    # Determine bounds on mixing weights.
    "mixing_weight_bounds": (0.1, 0.995),
    # Determine after how many re-discoveries of the currently best
    # local optimum the multistart optimization converges.
    "convergence.max_discoveries": 2,
    # Determine the maximum relative distance two parameter vectors
    # can have to be considered equal for convergence purposes:
    "convergence.relative_params_tolerance": 0.01,
    # Determine how many cores are used
    "n_cores": 1,
    # Determine which batch_evaluator is used:
    "batch_evaluator": "joblib",
    # Determine the batch size. It must be larger than n_cores.
    # Setting the batch size larger than n_cores allows to reproduce
    # the exact results of a highly parallel optimization on a smaller
    # machine.
    "batch_size": 4,
    # Set the random seed:
    "seed": None,
    # Set how errors are handled during the exploration phase:
    "exploration_error_handling": "continue",
    # Set how errors are handled during the optimization phase:
    "optimization_error_handling": "continue",
}
