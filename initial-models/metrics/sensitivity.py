import math
import numpy as np

print('Sensitivity module imported')

def generate_perturbation(value, feature_metadata, direction, numeric_displacement=0.1):

    # We use this to ensure zero values are perturbed
    ADDITIONAL_VALUE = 0.0001

    # If increasing - round up
    if (direction == 'up'):
        new_value = value * (1 + numeric_displacement) + ADDITIONAL_VALUE
        new_value = math.ceil(new_value)

    # If decreasing - round down
    else:
        new_value = value * (1 - numeric_displacement) - ADDITIONAL_VALUE
        new_value = math.floor(new_value)

    # Check no higher/lower than max/min values
    if ('max' in feature_metadata):
        new_value = min(new_value, feature_metadata['max'])
    if ('min' in feature_metadata):
        new_value = max(new_value, feature_metadata['min'])
    return new_value


def calculate_perturbation_numerical(instance, feature_metadata, explainer, original_explanation, numeric_displacement=0.1):

    feature_name = feature_metadata['name']

    max_explanation_difference = 0
    new_value = instance[feature_name]

    positive_copy = instance.copy(deep=True)
    positive_perturbed_value = generate_perturbation(
        positive_copy[feature_name], feature_metadata, 'up')
    positive_copy[feature_name] = positive_perturbed_value
    positive_explanation = explainer(positive_copy)
    positive_explanation_difference = np.linalg.norm(
        np.array(positive_explanation) - np.array(original_explanation))

    negative_copy = instance.copy(deep=True)
    negative_perturbed_value = generate_perturbation(
        negative_copy[feature_name], feature_metadata, 'down')
    negative_copy[feature_name] = negative_perturbed_value
    negative_explanation = explainer(negative_copy)
    negative_explanation_difference = np.linalg.norm(
        np.array(negative_explanation) - np.array(original_explanation))

    if positive_explanation_difference > max_explanation_difference:
        max_explanation_difference = positive_explanation_difference
        new_value = positive_perturbed_value

    if negative_explanation_difference > max_explanation_difference:
        max_explanation_difference = negative_explanation_difference
        new_value = negative_perturbed_value

    return max_explanation_difference, new_value


def calculate_perturbation_ordinal(instance, feature_metadata, explainer, original_explanation):

    feature_name = feature_metadata['name']
    value = instance[feature_name]

    # Get total number of values
    num_values = len(feature_metadata['values'])

    # Work out index of value
    value_index = 0
    for idx, val in enumerate(feature_metadata['values']):
        if val == value:
            value_index = idx
            break

    max_explanation_difference = 0
    new_value = instance[feature_name]

    for step in [-1, 1]:

        this_new_value = 0

        # Return value, looping to other end if required
        if value_index + step < 0:
            this_new_value = feature_metadata['values'][num_values - 1]

        elif value_index + step >= num_values:
            this_new_value = feature_metadata['values'][0]

        else:
            this_new_value = feature_metadata['values'][value_index + step]

        # Calculate new explanation and distance for the new value
        instance_copy = instance.copy(deep=True)
        instance_copy[feature_name] = this_new_value
        explanation = explainer(instance_copy)

        explanation_difference = np.linalg.norm(
            np.array(explanation) - np.array(original_explanation))

        # If this difference is highest, assign it (and the new value)
        if explanation_difference > max_explanation_difference:
            max_explanation_difference = explanation_difference
            new_value = this_new_value

    return max_explanation_difference, new_value


def calculate_perturbation_nominal(instance, feature_metadata, explainer, original_explanation):

    feature_name = feature_metadata['name']
    value = instance[feature_name]

    # Remove actual value from the list of values
    other_values = list(filter((lambda v: v != value),
                        feature_metadata['values']))

    # Set up starting values
    max_explanation_difference = 0
    new_value = instance[feature_name]

    # Loop through other possible values - see which causes the maximum perturbation
    for value in other_values:
        nominal_copy = instance.copy(deep=True)
        nominal_copy[feature_name] = value
        nominal_explanation = explainer(nominal_copy)

        nominal_explanation_difference = np.linalg.norm(
            np.array(nominal_explanation) - np.array(original_explanation))

        if nominal_explanation_difference > max_explanation_difference:
            max_explanation_difference = nominal_explanation_difference
            new_value = value

    return max_explanation_difference, new_value

# More sophisticated implementation of sensitivity


def calculate_sensitivity(explainer, instance_index, original_explanation, metadata_local, df_local, numeric_displacement=0.1, proportion_features_perturbed=0.1):

    # Filter our features not used
    used_features = list(filter(
        (lambda feat: feat['type'] != 'outcome' and feat['used'] == True), metadata_local))

    # Calculate how many features to perturb
    n = math.ceil(len(used_features) * proportion_features_perturbed)

    # Make a copy of the instance
    instance_copy = df_local.iloc[instance_index, :].copy(deep=True)

    # Loop through the number of features
    for iteration in range(n):

        # Keep track of the feature resulting in max difference
        max_difference = 0
        optimal_feature = ''
        new_value = 0

        # Loop through each 'used feature'
        for feature in used_features:

            # Apply the perturbations, varying the technique depending on the data type
            if feature['type'] == 'numerical':
                perturbation_difference, perturbation_value = calculate_perturbation_numerical(
                    instance_copy, feature, explainer, original_explanation)

            elif feature['type'] == 'ordinal':
                perturbation_difference, perturbation_value = calculate_perturbation_ordinal(
                    instance_copy, feature, explainer, original_explanation)

            elif feature['type'] == 'nominal':
                perturbation_difference, perturbation_value = calculate_perturbation_nominal(
                    instance_copy, feature, explainer, original_explanation)

            else:
                print('Error - not a recognised type')

            if perturbation_difference > max_difference:
                max_difference = perturbation_difference
                optimal_feature = feature['name']
                new_value = perturbation_value

        # Actually carry out the perturbation to the instance itself, and remove that feature from used_features so it isn't perturbed twice
        instance_copy[optimal_feature] = new_value
        used_features = list(
            filter((lambda feat: feat['name'] != optimal_feature), used_features))

    # Generate a new explanation based on the perturbed instance
    perturbed_explanation = explainer(instance_copy)

    # Calculate L2 distance between this and the original explanation
    difference = np.linalg.norm(
        np.array(perturbed_explanation) - np.array(original_explanation))

    return max_difference