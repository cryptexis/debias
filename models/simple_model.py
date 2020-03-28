import tensorflow as tf
from models.base_model import BaseModel


class SimpleClassifier(BaseModel):

    def __init__(self, predictor_model):

        super().__init__()
        # assigning predictor and discriminator models
        self.predictor = predictor_model

        # losses and optimizers
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.predictor_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

        self.metrics = [
            tf.keras.metrics.Mean(name='loss_mean'),
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy')
        ]

    @tf.function
    def _train_step(self, input_features, labels):
        """
        Executes a single training step
        :param input_features: input data in the shape [B, num_features]
        :param labels: Label column in the form [B, 1]
        :return: tuple (predicted_output, loss) for the current batch
        """
        with tf.GradientTape() as predictor_tape:
            # calculate prediction and the loss
            predicted_output = self.predictor(input_features, training=True)
            predictor_loss = self.loss(labels, predicted_output)

        # calculate and apply gradients
        gradients_of_predictor = predictor_tape.gradient(predictor_loss, self.predictor.trainable_variables)
        self.predictor_optimizer.apply_gradients(zip(gradients_of_predictor, self.predictor.trainable_variables))

        return tf.cast(tf.greater(predicted_output, 0.0), dtype=tf.int32), predictor_loss

