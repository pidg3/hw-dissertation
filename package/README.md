# Explainability robustness toolbox (exrt)

This package provides two metrics that can be used to measure the robustness of explanations of machine learning models - **infidelity** and **sensitivity**.

The metrics are based on those defined by Yeh et al. in their paper [On the (In)fidelity and Sensitivity of Explanations](https://papers.nips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf). The implementation is a little different, and allows for nominal data to be used, as well as any arbitary:

- Model, provided it has a `predict()` method that takes an array-like instance. 
- Explanation, provided it is an array-like set of numerical feature importances. 

Methods to assist in assembling the required metadata are also provided.

This work was part of an MSc project, full code for which (including analysis of the results on real datasets) can be found here: https://github.com/pidg3/hw-dissertation

## Infidelity

**Intuition**: if we know what features are most salient for a given model/instance combination, adjusting the values of these features should result in a large change in model output. If it does not, it implies the explanation is not faithful to the model, and therefore will return a larger infidelity value. 

**Usage**:
```
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
```

### Known issues

Features are adjusted to their baseline values. If the instance is close to these baseline values for the most salient features, this can result in an artificially low infidelity value. One possible future improvement could be taking the absolute value of `perturbation (dot) explanation - prediction_difference`, rather than the its square (as done by Yeh et al.).

## Sensitivity

**Intuition**: an explanation should not change substantially given a small perturbation to the instance being explained. If it does, it implies the explanation is lacking robustness. 

**Usage**:
```
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
```

### Known issues

Speed - the algorithm is currently rather naive and works through every possible combination of perturbations, calculating a new explanation for each. This can be improved slightly by reducing the `proportion_features_perturbed`, and setting `skip_zero_saliency_features=False` (particularly for datasets with large number of features).

There is an implicit assumption in this metric the underlying model is fully robust. If the model output itself changes significantly in response to a small perturbation, it is reasonable to also expect the explanation to change. 

## Metadata

Both metrics require a metadata object to be provided. This is a list of dictionaries, with the format below:

```
[
  {
    "name":"age",
    "type":"numerical",
    "used":true,
    "min":18,
    "max": 75,
    "index": 0,
    "baseline": 28
  },
  {
    "name":"job",
    "type":"nominal",
    "values":[
      "juggler",
      "lion-tamer",
      "human-cannonball",
    ],
    "used":true,
    "index": 1,
    "baseline": "juggler"
  },
  {
    "name":"day",
    "type":"ordinal",
    "values":[
      "mon",
      "tues",
      "wed",
      "thurs",
      "fri"
    ],
    "used":true,
    "index": 2,
    "baseline": "wed"
  }
]
```

Features should be divided into one of three types:
* `numerical` - can be integers or floats. Note that:
  * `min/max` values can optionally be provided. These will improve the accuracy of the sensitivity calculation.
  * `baseline` - should be the mean value.
* `nominal` - unordered strings. Note that:
  * `baseline` - should be the mode value (most frequently occuring).
* `ordinal` - ordered strings. Note that:
  * `baseline` - should be the median value.

Helper methods are provided to make this a litte easier: 

* `metadata.append_indices(metadata)` - returns a metadata object with `index` fields appended. 
* `metadata.append_baselines(metadata, dataframe)` - returns a metadata object with `baseline` fields appended, assuming the column headers of the dataframe match the feature names in the metadata object. 

Methods are also provided to help get useful data from the metadata object.

* `metadata.get_feature_names(metadata)` - returns a list of names.
* `metadata.get_feature_names_of_type(type, metadata)` - returns a list of names for numerical/nominal/ordinal features only.

## Full example

TODO...