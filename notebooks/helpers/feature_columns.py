import tensorflow as tf

def _make_zscaler(mean, std):
  def zscaler(col):
      return (col - mean)/std
  return zscaler

def get_normalization_parameters(dataframe, features):
  def calculate_params(column):
      mean = dataframe[column].mean()
      std = dataframe[column].std()
      return {'mean': mean, 'std': std}

  normalization_parameters = {}
  for column in features:
      normalization_parameters[column] = calculate_params(column)
  return normalization_parameters


def create_numeric_columns(dataframe, features, normalization_parameters, use_normalization=True):
    normalized_feature_columns = []
    for column_name in features:
        normalizer_fn = None
        if use_normalization:
            column_params = normalization_parameters[column_name]
            mean = column_params['mean']
            std = column_params['std']
            normalizer_fn = _make_zscaler(mean, std)
        normalized_feature_columns.append(tf.feature_column.numeric_column(column_name, normalizer_fn=normalizer_fn))
    return normalized_feature_columns

def create_categorical_columns(dataframe, features):
  categorical_feature_columns = []
  for col_name in features:
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        col_name, dataframe[col_name].unique())
    indicator_column = tf.feature_column.indicator_column(categorical_column)
    categorical_feature_columns.append(indicator_column)
  return categorical_feature_columns
