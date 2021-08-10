import math
import numpy as np


def _is_array_like(input):
    if isinstance(input, (list, np.ndarray)):
        return True
    return False


def _explainer_returns_correct_format(explainer, instance):
    if not callable(explainer):
        raise ValueError("Explainer not callable")
    try:
        result = explainer(instance)
    except:
        raise ValueError("Explainer function cannot handle instance")


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
    instance,
    feature_metadata,
    explainer,
    original_explanation,
    numeric_displacement=0.1,
):

    feature_index = feature_metadata["index"]

    max_explanation_difference = 0
    new_value = instance[feature_index]

    positive_copy = instance.copy()
    positive_perturbed_value = _generate_perturbation(
        positive_copy[feature_index], feature_metadata, "up"
    )
    positive_copy[feature_index] = positive_perturbed_value
    positive_explanation = explainer(positive_copy)
    positive_explanation_difference = np.linalg.norm(
        np.array(positive_explanation) - np.array(original_explanation)
    )

    negative_copy = instance.copy()
    negative_perturbed_value = _generate_perturbation(
        negative_copy[feature_index], feature_metadata, "down"
    )
    negative_copy[feature_index] = negative_perturbed_value
    negative_explanation = explainer(negative_copy)
    negative_explanation_difference = np.linalg.norm(
        np.array(negative_explanation) - np.array(original_explanation)
    )

    if positive_explanation_difference > max_explanation_difference:
        max_explanation_difference = positive_explanation_difference
        new_value = positive_perturbed_value

    if negative_explanation_difference > max_explanation_difference:
        max_explanation_difference = negative_explanation_difference
        new_value = negative_perturbed_value

    return max_explanation_difference, new_value


def _calculate_perturbation_ordinal(
    instance, feature_metadata, explainer, original_explanation
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

    max_explanation_difference = 0
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

        # Calculate new explanation and distance for the new value
        instance_copy = instance.copy()
        instance_copy[feature_index] = this_new_value
        explanation = explainer(instance_copy)

        explanation_difference = np.linalg.norm(
            np.array(explanation) - np.array(original_explanation)
        )

        # If this difference is highest, assign it (and the new value)
        if explanation_difference > max_explanation_difference:
            max_explanation_difference = explanation_difference
            new_value = this_new_value

    return max_explanation_difference, new_value


def _calculate_perturbation_nominal(
    instance, feature_metadata, explainer, original_explanation
):

    feature_index = feature_metadata["index"]
    value = instance[feature_index]

    # Remove actual value from the list of values
    other_values = list(filter((lambda v: v != value), feature_metadata["values"]))

    # Set up starting values
    max_explanation_difference = 0
    new_value = instance[feature_index]

    # Loop through other possible values - see which causes the maximum perturbation
    for value in other_values:
        nominal_copy = instance.copy()
        nominal_copy[feature_index] = value
        nominal_explanation = explainer(nominal_copy)

        nominal_explanation_difference = np.linalg.norm(
            np.array(nominal_explanation) - np.array(original_explanation)
        )

        if nominal_explanation_difference > max_explanation_difference:
            max_explanation_difference = nominal_explanation_difference
            new_value = value

    return max_explanation_difference, new_value


# More sophisticated implementation of sensitivity


def calculate_sensitivity(
    explainer,
    original_explanation,
    instance,
    metadata,
    numeric_displacement=0.1,
    proportion_features_perturbed=0.1,
    skip_zero_saliency_features=False,
):
    """
    calculate_sensitivity calculates a single numeric value for an explanation's sensitivity, without respect to the underlying model

    :param explainer: function that provides an explanation for a specific instance (in numpy format)
    :param original_explanation: array of numbers representing the original explanation
    :param instance: an array of numbers representing the input instance
    :param metadata: metadata dictionary in standard format
    :param numeric_displacement: how much (in percentage terms) to perturb numeric features by
    :param proportion_features_perturbed: how many features (in percentage terms, rounded up) to perturb
    :param skip_zero_saliency_features: whether to skip perturbing features with zero saliency value (i.e. we assume
      not important to the calculation)
    """

    # Check inputs array-like
    if not (_is_array_like(original_explanation) and _is_array_like(instance)):
        raise ValueError(
            "original_explanation and instance should be a list or np array"
        )

    # Check explainer is callable, and can handle instance withour erroring
    _explainer_returns_correct_format(explainer, instance)

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
        optimal_feature = ""
        new_value = 0

        # Loop through each 'used feature'
        for feature in used_features:
            if (
                skip_zero_saliency_features == False
                or instance_copy[feature["index"]] != 0
            ):
                # Apply the perturbations, varying the technique depending on the data type
                if feature["type"] == "numerical":
                    (
                        perturbation_difference,
                        perturbation_value,
                    ) = _calculate_perturbation_numerical(
                        instance_copy, feature, explainer, original_explanation
                    )

                elif feature["type"] == "ordinal":
                    (
                        perturbation_difference,
                        perturbation_value,
                    ) = _calculate_perturbation_ordinal(
                        instance_copy, feature, explainer, original_explanation
                    )

                elif feature["type"] == "nominal":
                    (
                        perturbation_difference,
                        perturbation_value,
                    ) = _calculate_perturbation_nominal(
                        instance_copy, feature, explainer, original_explanation
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

    # Generate a new explanation based on the perturbed instance
    perturbed_explanation = explainer(instance_copy)

    # Calculate L2 distance between this and the original explanation
    difference = np.linalg.norm(
        np.array(perturbed_explanation) - np.array(original_explanation)
    )

    return difference
