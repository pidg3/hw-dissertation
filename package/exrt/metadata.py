import copy


def get_feature_names(metadata):
    return list(map((lambda feat: feat["name"]), metadata))


def get_feature_names_of_type(type, metadata):
    features = list(
        filter((lambda feat: feat["type"] == type and feat["used"] == True), metadata)
    )
    names = list(map((lambda feat: feat["name"]), features))
    return names


def append_indices(metadata):
    metadata_copy = copy.deepcopy(metadata)
    for index, feat in enumerate(metadata_copy):
        feat["index"] = index
    return metadata_copy


def append_baselines(metadata, dataframe):
    metadata_copy = copy.deepcopy(metadata)
    for feat in metadata_copy:
        if feat["type"] == "numerical":
            feat["baseline"] = dataframe[feat["name"]].mean()

        elif feat["type"] == "ordinal":
            # Note we infer this from the metadata itself rather than the dataframe
            midpoint = round((len(feat["values"]) - 1) / 2)
            feat["baseline"] = feat["values"][midpoint]

        elif feat["type"] == "nominal":
            feat["baseline"] = dataframe[feat["name"]].mode()[0]

    return metadata_copy
