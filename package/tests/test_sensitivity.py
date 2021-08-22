import numpy as np
import pytest

from exrt.sensitivity import (
    calculate_sensitivity,
    _generate_perturbation,
    _calculate_perturbation_numerical,
    _calculate_perturbation_ordinal,
    _calculate_perturbation_nominal,
)


def mock_explainer(instance):
    return np.ones(len(instance))


mock_explanation = [0.25, 0.5]
mock_instance = [3, 5]
mock_metadata = [
    {
        "name": "x1",
        "type": "numerical",
        "used": True,
        "index": 0,
        "baseline": 1,
        "max": 5,
        "min": -5,
    },
    {
        "name": "x2",
        "type": "numerical",
        "used": True,
        "index": 1,
        "baseline": 1,
        "max": 5,
        "min": -5,
    },
]


def test_happy_path():
    result = calculate_sensitivity(
        mock_explainer, mock_explanation, mock_instance, mock_metadata
    )
    # Check we get a sensible, numerical result
    assert result > 0
    assert result < 1


# =========== Incorrect inputs ===========


def test_explanation_not_list_like():
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(mock_explainer, 5, mock_instance, mock_metadata)
    assert (
        str(exception.value)
        == "original_explanation and instance should be a list or np array"
    )


def test_instance_not_list_like():
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(mock_explainer, mock_explanation, 5, mock_metadata)
    assert (
        str(exception.value)
        == "original_explanation and instance should be a list or np array"
    )


def test_explainer_not_valid():
    # Not callable
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(5, mock_explanation, mock_instance, mock_metadata)
    assert str(exception.value) == "Explainer not callable"

    # Errors
    def explain_errors(instance):
        raise Exception("some internal error")

    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(
            explain_errors, mock_explanation, mock_instance, mock_metadata
        )
    assert str(exception.value) == "Explainer function cannot handle instance"


def test_bad_metadata():
    bad_metadata = [
        {"name": "x1", "type": "numerical", "used": True, "baseline": 1},
        {"name": "x2", "type": "numerical", "used": True, "index": 1, "baseline": 1},
    ]
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(
            mock_explainer, mock_explanation, mock_instance, bad_metadata
        )
    assert str(exception.value) == "Bad metadata - indexes not defined"


# =========== Helper methods ===========


def test_generate_perturbation():
    increased_value = _generate_perturbation(
        3, mock_metadata[0], "up", numeric_displacement=0.1
    )
    assert increased_value == 3.1
    decreased_value = _generate_perturbation(
        3, mock_metadata[0], "down", numeric_displacement=0.1
    )
    assert decreased_value == 2.9


def test_generate_perturbation_at_boundaries():
    increased_value = _generate_perturbation(
        3, mock_metadata[0], "up", numeric_displacement=100
    )
    assert increased_value == 5
    decreased_value = _generate_perturbation(
        3, mock_metadata[0], "down", numeric_displacement=100
    )
    assert decreased_value == -5


def test_calculate_perturbation_numerical():
    def explainer(instance):
        return [x * 2 for x in instance]

    max_explanation_difference, result = _calculate_perturbation_numerical(
        [3, 5], mock_metadata[0], explainer, [1, 2]
    )
    assert result == 3
