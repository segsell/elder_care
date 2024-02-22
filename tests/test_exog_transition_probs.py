from numpy.testing import assert_almost_equal as aaae

from elder_care.exogenous_processes.task_create_exog_processes import (
    exog_health_transition,
)


def test_exog_health_transition():
    params = {
        "medium_health": {
            "medium_health_age": 0.0304,
            "medium_health_age_squared": -1.31e-05,
            "medium_health_lagged_good_health": -1.155,
            "medium_health_lagged_medium_health": 0.736,
            "medium_health_lagged_bad_health": 1.434,
            "medium_health_constant": -1.550,
        },
        "bad_health": {
            "bad_health_age": 0.196,
            "bad_health_age_squared": -0.000885,
            "bad_health_lagged_good_health": -2.558,
            "bad_health_lagged_medium_health": -0.109,
            "bad_health_lagged_bad_health": 2.663,
            "bad_health_constant": -9.220,
        },
    }

    age = 50
    good_health, medium_health, bad_health = 1, 0, 0

    expected_prob_good, expected_prob_medium, expected_prob_bad = (
        0.76275676,
        0.22569606,
        0.01154716,
    )

    prob_good, prob_medium, prob_bad = exog_health_transition(
        age,
        good_health,
        medium_health,
        bad_health,
        params,
    )

    aaae(prob_good, expected_prob_good)
    aaae(prob_medium, expected_prob_medium)
    aaae(prob_bad, expected_prob_bad)
    aaae(prob_good + prob_medium + prob_bad, 1)
