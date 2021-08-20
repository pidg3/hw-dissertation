import math
import numpy as np


def _is_array_like(input):
    if isinstance(input, (list, np.ndarray)):
        return True
    return False


def _model_returns_correct_format(model, instance):
    if not callable(model):
        raise ValueError("model not callable")
    try:
        result = model(instance)
    except:
        raise ValueError("model function cannot handle instance")


def _generate_perturbation(
    value, feature_metadata, direction, numeric_displacement=0.1
):

    # If increasing
    if direction == "up":
        new_value = value + numeric_displacement * feature_metadata["baseline"]

    # If decreasing
    else:
        new_value = value - numeric_displacement * feature_metadata["baseline"]

    # Check no higher/lower than max/min values
    if "max" in feature_metadata:
        new_value = min(new_value, feature_metadata["max"])
    if "min" in feature_metadata:
        new_value = max(new_value, feature_metadata["min"])
    return new_value


def _calculate_perturbation_numerical(
    instance, feature_metadata, model, original_prediction, numeric_displacement=0.1,
):

    feature_index = feature_metadata["index"]

    max_prediction_difference = 0
    new_value = instance[feature_index]

    positive_copy = instance.copy()
    positive_perturbed_value = _generate_perturbation(
        positive_copy[feature_index], feature_metadata, "up"
    )
    positive_copy[feature_index] = positive_perturbed_value
    positive_prediction = model(positive_copy)
    positive_prediction_difference = abs(positive_prediction - original_prediction)

    negative_copy = instance.copy()
    negative_perturbed_value = _generate_perturbation(
        negative_copy[feature_index], feature_metadata, "down"
    )
    negative_copy[feature_index] = negative_perturbed_value
    negative_prediction = model(negative_copy)
    negative_prediction_difference = abs(negative_prediction - original_prediction)

    if positive_prediction_difference > max_prediction_difference:
        max_prediction_difference = positive_prediction_difference
        new_value = positive_perturbed_value

    if negative_prediction_difference > max_prediction_difference:
        max_prediction_difference = negative_prediction_difference
        new_value = negative_perturbed_value

    return max_prediction_difference, new_value


def _calculate_perturbation_ordinal(
    instance, feature_metadata, model, original_prediction
):

    feature_index = feature_metadata["index"]
    value = instance[feature_index]

    # Get total number of values
    num_values = len(feature_metadata["values"])

    # Work out index of value
    value_index = 0
    for idx, val in enumerate(feature_metadata["values"]):
        if val == value:
            value_index = idx
            break

    max_prediction_difference = 0
    new_value = instance[feature_index]

    for step in [-1, 1]:

        this_new_value = 0

        # Return value, looping to other end if required
        if value_index + step < 0:
            this_new_value = feature_metadata["values"][num_values - 1]

        elif value_index + step >= num_values:
            this_new_value = feature_metadata["values"][0]

        else:
            this_new_value = feature_metadata["values"][value_index + step]

        # Calculate new prediction and difference for the new value
        instance_copy = instance.copy()
        instance_copy[feature_index] = this_new_value
        prediction = model(instance_copy)
        prediction_difference = abs(prediction - original_prediction)

        # If this difference is highest, assign it (and the new value)
        if prediction_difference > max_prediction_difference:
            max_prediction_difference = prediction_difference
            new_value = this_new_value

    return max_prediction_difference, new_value


def _calculate_perturbation_nominal(
    instance, feature_metadata, model, original_prediction
):

    feature_index = feature_metadata["index"]
    value = instance[feature_index]

    # Remove actual value from the list of values
    other_values = list(filter((lambda v: v != value), feature_metadata["values"]))

    # Set up starting values
    max_prediction_difference = 0
    new_value = instance[feature_index]

    # Loop through other possible values - see which causes the maximum perturbation
    for value in other_values:
        nominal_copy = instance.copy()
        nominal_copy[feature_index] = value
        nominal_prediction = model(nominal_copy)

        nominal_prediction_difference = abs(nominal_prediction - original_prediction)

        if nominal_prediction_difference > max_prediction_difference:
            max_prediction_difference = nominal_prediction_difference
            new_value = value

    return max_prediction_difference, new_value


# More sophisticated implementation of sensitivity
def calculate_robustness(
    model,
    original_prediction,
    instance,
    metadata,
    numeric_displacement=0.1,
    proportion_features_perturbed=0.1,
):

    # Check inputs array-like
    if not _is_array_like(instance):
        raise ValueError("instance should be a list or np array")

    # Check explainer is callable, and can handle instance withour erroring
    _model_returns_correct_format(model, instance)

    # Check metadata has all required fields
    for feat in metadata:
        if feat["used"] == True:
            if "index" not in feat:
                raise ValueError("Bad metadata - indexes not defined")
            if "name" not in feat:
                raise ValueError("Bad metadata - names not defined")
            if "type" not in feat:
                raise ValueError("Bad metadata - types not defined")
            if "baseline" not in feat:
                raise ValueError("Bad metadata - baselines not defined")

    # Filter our features not used
    used_features = list(
        filter(
            (lambda feat: feat["type"] != "outcome" and feat["used"] == True), metadata
        )
    )

    # Calculate how many features to perturb
    n = math.ceil(len(used_features) * proportion_features_perturbed)

    # Make a copy of the instance
    instance_copy = instance.copy()

    # Loop through the number of features
    for iteration in range(n):

        # Keep track of the feature resulting in max difference
        max_difference = 0
        new_value = 0

        # Loop through each 'used feature'
        for feature in used_features:
            # Apply the perturbations, varying the technique depending on the data type
            if feature["type"] == "numerical":
                (
                    perturbation_difference,
                    perturbation_value,
                ) = _calculate_perturbation_numerical(
                    instance_copy, feature, model, original_prediction
                )

            elif feature["type"] == "ordinal":
                (
                    perturbation_difference,
                    perturbation_value,
                ) = _calculate_perturbation_ordinal(
                    instance_copy, feature, model, original_prediction
                )

            elif feature["type"] == "nominal":
                (
                    perturbation_difference,
                    perturbation_value,
                ) = _calculate_perturbation_nominal(
                    instance_copy, feature, model, original_prediction
                )

            else:
                print("Error - not a recognised type")

            if perturbation_difference > max_difference:
                max_difference = perturbation_difference
                optimal_feature_index = feature["index"]
                new_value = perturbation_value

        # Actually carry out the perturbation to the instance itself, and remove that feature from used_features so it isn't perturbed twice
        instance_copy[optimal_feature_index] = new_value
        used_features = list(
            filter((lambda feat: feat["index"] != optimal_feature_index), used_features)
        )

    # Generate a new prediction based on the perturbed instance
    perturbed_prediction = model(instance_copy)

    # Calculate difference between this and the original explanation
    difference = abs(perturbed_prediction - original_prediction)

    # Return the inverse - high robustness = low sensitivity
    return 1 / difference
