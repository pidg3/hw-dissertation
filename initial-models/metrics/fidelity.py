import copy
import numpy as np

print('Fidelity module imported')

def calculate_fidelity(explanation, model, dataframe, metadata, instance_index, num_perturbed_features=2, nominal_perturbation_constant=1):

    # Make copies of the instance and metadata
    instance_original = dataframe.iloc[instance_index, :].copy(deep=True)
    instance_perturbed = dataframe.iloc[instance_index, :].copy(deep=True)
    metadata_copy = copy.deepcopy(metadata)

    # Work out an ordering for nominal values
    for feature in metadata_copy:
        if feature['type'] == 'nominal':
            outputs = []
            instance_nominal_ranking = dataframe.iloc[instance_index, :].copy(
                deep=True)
            for possible_value in feature['values']:
                instance_nominal_ranking[feature['index']] = possible_value
                outputs.append({
                    'value': possible_value,
                    'output': model([instance_nominal_ranking.to_numpy()])
                })
            sorted_outputs = sorted(outputs, key=(lambda feat: feat['output']))
            feature['values'] = list(
                map((lambda item: item['value']), sorted_outputs))
            midpoint = round((len(feature['values']) - 1) / 2)
            feature['baseline'] = feature['values'][midpoint]

    # Get our top n feature names
    explanation_with_indexes = []
    index = 0
    for feat in explanation[0]:
        explanation_with_indexes.append({
            'index': index,
            'value': feat
        })
        index += 1

    sorted_explanations = sorted(
        explanation_with_indexes, key=(lambda feat: -abs(feat['value'])))
    top_explanations = sorted_explanations[0:num_perturbed_features]

    # Make a copy of the instance
    instance_original = dataframe.iloc[instance_index, :].copy(deep=True)
    instance_perturbed = dataframe.iloc[instance_index, :].copy(deep=True)

    # Reassign these features to their baseline values
    for feat in top_explanations:
        idx = feat['index']
        instance_perturbed[idx] = metadata_copy[idx]['baseline']

    # Calculate the resulting perturbation
    instance_original_np = instance_original.to_numpy()
    instance_perturbed_np = instance_perturbed.to_numpy()

    perturbations = []

    for index, (original_value, perturbed_value) in enumerate(zip(instance_original_np, instance_perturbed_np)):
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
    infidelity = np.dot(perturbations, explanation[0]) - ((model(
        [instance_original_np])[0] - model([instance_perturbed_np])[0]) ** 2)
    return 1 / infidelity[0]
