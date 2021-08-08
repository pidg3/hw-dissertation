import numpy as np
import pytest
from infidelity import calculate_infidelity

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
    calculate_infidelity(mock_explanation, mock_model, mock_instance, mock_metadata)

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
