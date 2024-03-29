import jax
import jax.numpy as jnp
import numpy as np


def draw_initial_states(initial_conditions, initial_wealth, n_agents, seed):
    """Draw initial resources and states for the simulation.

    mother_health = get_initial_share_three(     initial_conditions,
    ["mother_good_health", "mother_medium_health", "mother_bad_health"], ) father_health
    = get_initial_share_three(     initial_conditions,     ["father_good_health",
    "father_medium_health", "father_bad_health"], )

    informal_care = get_initial_share_two(initial_conditions, "share_informal_care")

    """
    n_choices = 12

    employment = get_initial_share_three(
        initial_conditions,
        ["share_not_working", "share_part_time", "share_full_time"],
    )

    informal_care = jnp.array([1, 0])  # agents start with no informal care provided
    formal_care = jnp.array([1, 0])  # agents start with no formal care provided

    informal_formal = jnp.outer(informal_care, formal_care)
    lagged_choice_probs = jnp.outer(employment, informal_formal).flatten()

    high_educ = get_initial_share_two(initial_conditions, "share_high_educ")
    has_sibling = get_initial_share_two(initial_conditions, "share_has_sister")

    mother_alive = get_initial_share_two(initial_conditions, "share_mother_alive")

    initial_resources = draw_random_sequence_from_array(
        seed,
        initial_wealth,
        n_agents=n_agents,
    )

    initial_states = {
        "period": jnp.zeros(n_agents, dtype=np.uint8),
        "lagged_choice": draw_random_array(
            seed=seed - 1,
            n_agents=n_agents,
            values=jnp.arange(n_choices),
            probabilities=lagged_choice_probs,
        ).astype(np.uint8),
        "high_educ": draw_random_array(
            seed=seed - 2,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=high_educ,
        ).astype(np.uint8),
        "has_sibling": draw_random_array(
            seed=seed - 3,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=has_sibling,
        ).astype(np.uint8),
        "experience": draw_from_discrete_normal(
            seed=seed - 4,
            n_agents=n_agents,
            mean=initial_conditions["experience_mean"],
            std_dev=initial_conditions["experience_std"],
        ),
        "part_time_offer": jnp.ones(n_agents, dtype=np.uint8),
        "full_time_offer": jnp.ones(n_agents, dtype=np.uint8),
        "care_demand": jnp.zeros(n_agents, dtype=np.uint8),
        "mother_alive": draw_random_array(
            seed=seed - 6,
            n_agents=n_agents,
            values=jnp.array([0, 1]),
            probabilities=mother_alive,
        ).astype(np.uint8),
    }

    return initial_resources, initial_states


def get_initial_share_three(initial_conditions, shares):
    return jnp.asarray(initial_conditions.loc[shares]).ravel()


def get_initial_share_two(initial_conditions, var):
    share_yes = initial_conditions.loc[var]
    return jnp.asarray([1 - share_yes, share_yes]).ravel()


def draw_random_sequence_from_array(seed, arr, n_agents):
    """Draw a random sequence from an array.

    rand = draw_random_sequence_from_array(     seed=2024,     n_agents=10_000,
    arr=jnp.array(initial_wealth), )

    """
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, arr, shape=(n_agents,), replace=True)


def draw_random_array(seed, n_agents, values, probabilities):
    """Draw a random array with given probabilities.

    Usage:

    seed = 2024
    n_agents = 10_000

    # Parameters
    values = jnp.array([-1, 0, 1, 2])  # Values to choose from
    probabilities = jnp.array([0.3, 0.3, 0.2, 0.2])  # Corresponding probabilities

    table(pd.DataFrame(random_array)[0]) / 1000

    """
    key = jax.random.PRNGKey(seed)
    return jax.random.choice(key, values, shape=(n_agents,), p=probabilities)


def draw_from_discrete_normal(seed, n_agents, mean, std_dev):
    """Draw from discrete normal distribution."""
    key = jax.random.PRNGKey(seed)

    sample_standard_normal = jax.random.normal(key, (n_agents,))

    # Scaling and shifting to get the desired mean and standard deviation, then rounding
    return jnp.round(mean + std_dev * sample_standard_normal).astype(jnp.uint16)
