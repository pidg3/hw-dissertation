import tensorflow as tf

print('Dataset converter module imported')

# A utility method to create a tf.data dataset from a Pandas Dataframe
# From: https://www.tensorflow.org/tutorials/structured_data/feature_columns
def convert_df_dataset(dataframe, y_label, shuffle=True, batch_size=100):
  dataframe = dataframe.copy()
  labels = dataframe.pop(y_label)
  # Just passing in a single argument here (tuple)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def convert_dataframe_numpy(dataframe, index):
    return [dataframe.iloc[index, :].to_numpy()]

def convert_numpy_tensor(instance, feature_names):
    data_for_tensor = {}
    for index, name in enumerate(feature_names):
        values = []
        for val in instance:
            values.append(val[index])
        data_for_tensor[name] = values
    tensor = {name: tf.convert_to_tensor(value)
              for name, value in data_for_tensor.items()}
    return tensor
