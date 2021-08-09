import numpy as np
import pytest
from sensitivity import calculate_sensitivity, _generate_perturbation, _calculate_perturbation_numerical, _calculate_perturbation_ordinal, _calculate_perturbation_nominal

def mock_explainer(instance):
    return np.ones(len(instance))
mock_explanation = [0.25, 0.5]
mock_instance = [3, 5]
mock_metadata = [
    {
        'name': 'x1',
        'type': 'numerical',
        'used': True,
        'index': 0,
        'baseline': 1
    },
    {
        'name': 'x2',
        'type': 'numerical',
        'used': True,
        'index': 1,
        'baseline': 1
    }
]

def test_happy_path():
    result = calculate_sensitivity(mock_explainer, mock_explanation, mock_instance, mock_metadata)
    # Check we get a sensible, numerical result
    assert result > 0
    assert result < 1

# =========== Incorrect inputs ===========

def test_explanation_not_list_like():
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(mock_explainer, 5, mock_instance, mock_metadata)
    assert str(exception.value) == 'original_explanation and instance should be a list or np array'

def test_instance_not_list_like():
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(mock_explainer, mock_explanation, 5, mock_metadata)
    assert str(exception.value) == 'original_explanation and instance should be a list or np array'

def test_explainer_not_valid():
    # Not callable
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(5, mock_explanation, mock_instance, mock_metadata)
    assert str(exception.value) == 'Explainer not callable'

    # Errors
    def explain_errors(instance):
        raise Exception('some internal error')
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(explain_errors, mock_explanation, mock_instance, mock_metadata)
    assert str(exception.value) == 'Explainer function cannot handle instance'

def test_bad_metadata():
    bad_metadata = [
        {
            'name': 'x1',
            'type': 'numerical',
            'used': True,
            'baseline': 1
        },
        {
            'name': 'x2',
            'type': 'numerical',
            'used': True,
            'index': 1,
            'baseline': 1
        }
    ]
    with pytest.raises(ValueError) as exception:
        calculate_sensitivity(mock_explainer, mock_explanation, mock_instance, bad_metadata)
    assert str(exception.value) == 'Bad metadata - indexes not defined'
