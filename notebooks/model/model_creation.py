import tensorflow as tf
from tensorflow.keras import layers

def create_fit_model(input_data, input_layer, metrics, epochs=10,
                     num_layers=3, dropout=True, verbose=1):

    # Add input layer
    layer_builder = [input_layer]

    # Add hidden layer, depending on hyperparameters
    for layer in range(num_layers):
        layer_builder.append(layers.Dense(128, activation='relu'))
        if dropout == True:
            layer_builder.append(layers.Dropout(0.2))

    # Add output layer
    layer_builder.append(layers.Dense(1, activation='sigmoid'))
    model = tf.keras.Sequential(layer_builder)

    # Compile and train our model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=metrics)

    # Note that sequential model warnings can be ignored: https://github.com/tensorflow/recommenders/issues/188
    history = model.fit(input_data, epochs=epochs, verbose=verbose)
    return model, history
