import numpy as np
import pytest
from infidelity import calculate_infidelity, _order_nominal_values, _get_top_explanation_values, _calculate_perturbations

class MockModel():
    def predict(self, instance):
        return 1

mock_explanation = np.array([-0.5, 0.5])
mock_model = MockModel()
mock_instance = [0, 1]
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
    result = calculate_infidelity(mock_explanation, mock_model, mock_instance, mock_metadata)
    assert result == 0.25

# =========== Incorrect inputs ===========

def test_explanation_not_list_like():
    with pytest.raises(ValueError) as exception:
        calculate_infidelity(10, mock_model, mock_instance, mock_metadata)
    assert str(exception.value) == 'Explanation and instance should be a list or np array'

def test_instance_not_list_like():
    with pytest.raises(ValueError) as exception:
        calculate_infidelity(mock_explanation, mock_model, 10, mock_metadata)
    assert str(exception.value) == 'Explanation and instance should be a list or np array'

def test_model_has_no_predict_func():
    class MockModelNoPredict():
        def foo(self):
            return 1
    with pytest.raises(ValueError) as exception:
        calculate_infidelity(mock_explanation, MockModelNoPredict(), mock_instance, mock_metadata)
    assert str(exception.value) == 'Model does not have predict() method'

    class MockModelPredictError():
        def predict(self, instance):
            raise Exception('Some internal error')
    with pytest.raises(ValueError) as exception:
        calculate_infidelity(mock_explanation, MockModelPredictError(), mock_instance, mock_metadata)
    assert str(exception.value) == 'Model predict() method cannot handle instance'

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
        calculate_infidelity(mock_explanation, mock_model, mock_instance, bad_metadata)
    assert str(exception.value) == 'Bad metadata - indexes not defined'

def test_bad_inconsistent_num_features():
    explanation_one_feature = [1]
    with pytest.raises(ValueError) as exception:
        calculate_infidelity(explanation_one_feature, mock_model, mock_instance, mock_metadata)
    assert str(exception.value) == 'Explanation, instance and metadata (used features) must be equal lengths'

    instance_one_feature = [1]
    with pytest.raises(ValueError) as exception:
        calculate_infidelity(mock_explanation, mock_model, instance_one_feature, mock_metadata)
    assert str(exception.value) == 'Explanation, instance and metadata (used features) must be equal lengths'

    metadata_one_feature = [
    {
        'name': 'x1',
        'type': 'numerical',
        'used': True,
        'index': 0,
        'baseline': 1
    }]
    with pytest.raises(ValueError) as exception:
        calculate_infidelity(mock_explanation, mock_model, mock_instance, metadata_one_feature)
    assert str(exception.value) == 'Explanation, instance and metadata (used features) must be equal lengths'

# =========== Helper methods ===========

def test_order_nominal_values(): 
    mock_metadata = [{
        'name': 'x1',
        'type': 'nominal',
        'used': True,
        'index': 0,
        'values': ['bananas', 'apples', 'pears'],
        'baseline': 'pears'
    }]
    mock_instance = ['peaches']
    class MockModel():
        def predict(self, instance):
            if instance == ['apples']:
                return -1
            if instance == ['bananas']:
                return 0
            if instance == ['pears']:
                return 1
    result = _order_nominal_values(mock_metadata, mock_instance, MockModel())
    expected_result = [{
        'name': 'x1',
        'type': 'nominal',
        'used': True,
        'index': 0,
        'values': ['apples', 'bananas', 'pears'], # Updated
        'baseline': 'bananas' # Updated

    }]
    assert result == expected_result

def test_get_top_explanation_values(): 
    mock_explanation = [-1, -5, 4, 0, 1, 3, 10]
    assert _get_top_explanation_values(mock_explanation, 2) == [
        {
            'index': 6,
            'value': 10
        },
        {
            'index': 1,
            'value': -5
        }
    ]

# Metadata has already been converted to 'ordered nominal' format at this point
def test_calculate_perturbation_numerical():
    instance_original = [5]
    instance_perturned = [10]
    mock_metadata = [
        {
            'name': 'x1',
            'type': 'numerical',
            'used': True,
            'index': 0,
            'baseline': 2
        }
    ]
    result = _calculate_perturbations(instance_original, instance_perturned, mock_metadata)
    # difference / baseline
    assert result == [2.5]

def test_calculate_perturbation_categorical():
    instance_original = ['apples']
    instance_perturned = ['bananas']
    mock_metadata = [{
        'name': 'x1',
        'type': 'nominal',
        'used': True,
        'index': 0,
        'values': ['apples', 'bananas', 'pears', 'kiwis', 'grapes', 'pineapples', 'melons'],
        'baseline': 'kiwis'

    }]
    result = _calculate_perturbations(instance_original, instance_perturned, mock_metadata)
    # original_index = 0
    # perturbed_index = 1
    # differernce = -1
    # abs(-1 / 6)
    assert result == [1 / 7]
