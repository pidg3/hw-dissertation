import tensorflow as tf

print('Dataset converter module imported')

# A utility method to create a tf.data dataset from a Pandas Dataframe
# From: https://www.tensorflow.org/tutorials/structured_data/feature_columns
def df_to_dataset(dataframe, y_label, shuffle=True, batch_size=100):
  dataframe = dataframe.copy()
  labels = dataframe.pop(y_label)
  # Just passing in a single argument here (tuple)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds
