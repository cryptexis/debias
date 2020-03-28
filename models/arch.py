import tensorflow as tf


def make_predictor_model(feature_layer):
    """
    Defines the predictor architecture
    :param feature_layer: TF dense_feature layer
    :return:
    """

    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(1)
    ])

    return model


def make_discriminator_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(3,))
    ])

    return model
