import copy
import numpy as np
from random import uniform
from exrt.metadata import get_feature_names


def _is_array_like(input):
    if isinstance(input, (list, np.ndarray)):
        return True
    return False


def _predict_returns_correct_format(instance, model):
    if not (hasattr(model, "predict") and callable(model.predict)):
        raise ValueError("Model does not have predict() method")
    try:
        result = model.predict(instance)
    except:
        raise ValueError("Model predict() method cannot handle instance")


def _order_nominal_values(metadata, instance, model):
    metadata_copy = copy.deepcopy(metadata)
    for feature in metadata_copy:
        if feature["type"] == "nominal":
            outputs = []
            instance_nominal_ranking = instance.copy()
            for possible_value in feature["values"]:
                instance_nominal_ranking[feature["index"]] = possible_value
                outputs.append(
                    {
                        "value": possible_value,
                        "output": model.predict(instance_nominal_ranking),
                    }
                )
            sorted_outputs = sorted(outputs, key=(lambda feat: feat["output"]))
            feature["values"] = list(map((lambda item: item["value"]), sorted_outputs))
            midpoint = round((len(feature["values"]) - 1) / 2)
            feature["baseline"] = feature["values"][midpoint]
    return metadata_copy


def _get_top_explanation_values(explanation, num_baselined_features):
    explanation_with_indexes = []
    index = 0
    for feat in explanation:
        explanation_with_indexes.append({"index": index, "value": feat})
        index += 1
    sorted_explanations = sorted(
        explanation_with_indexes, key=(lambda feat: -abs(feat["value"]))
    )
    return sorted_explanations[0:num_baselined_features]


def _calculate_perturbations(instance_original, instance_perturbed, metadata):
    perturbations = []

    for index, (original_value, perturbed_value) in enumerate(
        zip(instance_original, instance_perturbed)
    ):
        # Work out type and set default value
        feat_type = metadata[index]["type"]
        perturbation = 0

        # Numeric - relative change, normalised by baseline
        if feat_type == "numerical":
            perturbation = abs(
                (original_value - perturbed_value) / metadata[index]["baseline"]
            )

        # Ordinal - difference in position, divided by total number of possible values for the feature
        elif feat_type == "ordinal" or feat_type == "nominal":
            original_index = metadata[index]["values"].index(original_value)
            perturbed_index = metadata[index]["values"].index(perturbed_value)
            perturbation = abs(
                (original_index - perturbed_index) / len(metadata[index]["values"])
            )

        perturbations.append(perturbation)
    return perturbations


def calculate_infidelity(
    explanation, model, instance, metadata, num_baselined_features=2
):
    """
    calculate_infidelity calculates a single numeric value for an explanation's infidelity with respect to some model
    
    Values are bounded from zero (huge change in model output) to +inf (zero change in model output)

    :param explanation: an array of numbers representing feature importances
    :param model: a model that provides a predict() function to generate an output prediction
    :param instance: an array of numbers representing the input instance
    :param metadata: metadata dictionary in standard format
    :param num_baselined_features: how many features to set to their baseline value before measuring model output
    """

    # Check inputs array-like
    if not (_is_array_like(explanation) and _is_array_like(instance)):
        raise ValueError("Explanation and instance should be a list or np array")

    # Check consistent array lengths
    metadata_used_only = [feat for feat in metadata if feat["used"] == True]
    if len(explanation) != len(instance) or len(explanation) != len(metadata_used_only):
        raise ValueError(
            "Explanation, instance and metadata (used features) must be equal lengths"
        )

    # Check model has predict method, and can handle instance withour erroring
    _predict_returns_correct_format(instance, model)

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

    feature_names = get_feature_names(metadata)
    metadata_copy = copy.deepcopy(metadata)
    abs_explanation = [abs(ele) for ele in explanation]

    # Work out an ordering for nominal values
    metadata_copy = _order_nominal_values(metadata_copy, instance, model)

    # Get the top n feature names, with most salient features as per the explanation (in absolute terms)
    top_explanations = _get_top_explanation_values(explanation, num_baselined_features)

    # Baseline each of our top features and measure the cumlalative total of infidelity
    # nb - important to do this for individual features, as otherwise we could baseline two features with
    # ... positive and negative values, which would largely cancel out each others' impact on output
    cumulative_infidelity = 0

    for feat in top_explanations:

        # Make copies of the instance
        instance_original = instance.copy()
        instance_perturbed = instance.copy()

        # Reassign the individual feature to its baseline value
        idx = feat["index"]
        instance_perturbed[idx] = metadata_copy[idx]["baseline"]

        # Calculate the array of resulting perturbations
        perturbations = _calculate_perturbations(
            instance_original, instance_perturbed, metadata_copy
        )

        # Apply the formula to calculate infidelity
        infidelity = np.square(
            np.dot(perturbations, explanation)
            - ((model.predict(instance_original) - model.predict(instance_perturbed)))
        )

        cumulative_infidelity += infidelity
    return cumulative_infidelity
