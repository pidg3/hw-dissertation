import copy
import numpy as np
from random import uniform
from model.dataset_converter import convert_numpy_tensor
from model.metadata import get_feature_names

print('Infidelity module imported')


def calculate_infidelity(explanation, model, instance, metadata, num_baselined_features=2):
    """
    calculate_infidelity calculates a single numeric value for an explanation's infidelity with respect to some model
    
    Values are bounded from zero (huge change in model output) to +inf (zero change in model output)

    :param explanation: an array of numbers representing feature importances TODO: should just be 1D array
    :param model: a model that provides a predict() function to generate an output prediction
    :param instance: an array of numbers representing the input instance
    :param metadata: metadata dictionary in standard format
    :param num_baselined_features: how many features to set to their baseline value before measuring model output
    :param
    """

    feature_names = get_feature_names(metadata)
    metadata_copy = copy.deepcopy(metadata)
    abs_explanation = [abs(ele) for ele in explanation]

    # Work out an ordering for nominal values
    for feature in metadata_copy:
        if feature['type'] == 'nominal':
            outputs = []
            instance_nominal_ranking = instance.copy()
            for possible_value in feature['values']:
                instance_nominal_ranking[feature['index']] = possible_value
                outputs.append({
                    'value': possible_value,
                    'output': model.predict(instance_nominal_ranking)
                })
            sorted_outputs = sorted(outputs, key=(lambda feat: feat['output']))
            feature['values'] = list(
                map((lambda item: item['value']), sorted_outputs))
            midpoint = round((len(feature['values']) - 1) / 2)
            feature['baseline'] = feature['values'][midpoint]

    # Get our top n feature names (in absolute terms)
    explanation_with_indexes = []
    index = 0
    for feat in explanation:
        explanation_with_indexes.append({
            'index': index,
            'value': feat
        })
        index += 1
    sorted_explanations = sorted(
        explanation_with_indexes, key=(lambda feat: -abs(feat['value'])))
    top_explanations = sorted_explanations[0:num_baselined_features]

    # Baseline each of our top features and measure the cumlalative total of infidelity
    # nb - important to do this for individual features, as otherwise we could baseline two features with
    # ... positive and negative values, which would largely cancel out each others' impact on output

    cumulative_infidelity = 0

    for feat in top_explanations:

        # Make copies of the instance
        instance_original = instance.copy()
        instance_perturbed = instance.copy()

        # Reassign the individual feature to its baseline value
        idx = feat['index']
        instance_perturbed[idx] = metadata_copy[idx]['baseline']

        # Calculate the resulting perturbation
        perturbations = []

        for index, (original_value, perturbed_value) in enumerate(zip(instance_original, instance_perturbed)):
            # Work out type and set default value
            feat_type = metadata_copy[index]['type']
            perturbation = 0

            # Numeric - relative change, normalised by baseline
            if feat_type == 'numerical':
                perturbation = abs(
                    (original_value - perturbed_value) / metadata_copy[index]['baseline'])

            # Ordinal - difference in position, divided by total number of possible values for the feature
            elif feat_type == 'ordinal' or feat_type == 'nominal':
                original_index = metadata_copy[index]['values'].index(
                    original_value)
                perturbed_index = metadata_copy[index]['values'].index(
                    perturbed_value)
                perturbation = abs(
                    (original_index - perturbed_index) / len(metadata_copy[index]['values']))

            perturbations.append(perturbation)

        # Apply the formula to calculate infidelity
        infidelity = np.square(
            np.dot(perturbations, explanation)
            -
            ((model.predict(instance_original) - model.predict(instance_perturbed)))
        )

        cumulative_infidelity += infidelity
    return cumulative_infidelity
