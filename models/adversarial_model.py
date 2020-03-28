import tensorflow as tf
import numpy as np
from utils.data import convert_categorical
from models.base_model import BaseModel


class Discriminator:

    def __init__(self, discriminator_model, protected_variable):

        self.model = discriminator_model
        self.protected_variable = protected_variable


class FairClassifier(BaseModel):

    def __init__(self, predictor_model, discriminator_model: Discriminator, hyper_parameters=None):

        # assigning predictor and discriminator models
        self.predictor = predictor_model
        self.discriminator = discriminator_model

        # losses and optimizers
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.cosine_loss = tf.keras.losses.CosineSimilarity()
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

        self.hyper_parameters = hyper_parameters if hyper_parameters is not None else {}

    def __predictor_gradient(self, gradients_of_predictor_pred_loss, gradients_of_predictor_disc_loss):
        """
        Calculate the final form of the gradient of the predictor network
        :param gradients_of_predictor_pred_loss: gradient of parameters based on the loss from predictor network
        :param gradients_of_predictor_disc_loss: gradient of parameters based on the loss from discriminator network
        :return:
        """
        gradients_of_predictor = []
        num_gradients = len(gradients_of_predictor_disc_loss)
        for i in range(num_gradients):
            # weighted gradient coming from the discriminator
            alpha = self.hyper_parameters.get("alpha", 1.0)
            disc_term = alpha*gradients_of_predictor_disc_loss[i]
            # projection of the gradient onto the discriminator gradient
            cosine_term = self.cosine_loss(gradients_of_predictor_pred_loss[i], gradients_of_predictor_disc_loss[i])
            proj_term = (cosine_term*tf.norm(gradients_of_predictor_pred_loss[i])*gradients_of_predictor_disc_loss[i])/\
                        tf.norm(gradients_of_predictor_disc_loss[i])

            # final form of the gradient
            gradients_of_predictor.append(gradients_of_predictor_pred_loss[i] - proj_term - disc_term)

        return gradients_of_predictor

    @tf.function
    def _train_step(self, input_features, labels):

        with tf.GradientTape() as predictor_tape, tf.GradientTape(persistent=True) as disc_tape:

            # predicting the label
            predictor_output = self.predictor(input_features, training=True)
            predictor_loss = self.loss(labels, predictor_output)

            # creating input for the discriminator
            labels = tf.cast(labels, dtype=tf.float32)
            # (
            s = (1.0 + np.abs(self.hyper_parameters.get('c', 1.0)))*predictor_output
            discriminator_input = tf.squeeze(tf.stack([s, s*labels, s*(1.0 - labels)], axis=1))

            # predicting the protected_variable
            discriminator_ouput = self.discriminator.model(discriminator_input, training=True)
            # converting protected variable into target column
            protected_feature = tf.keras.layers.DenseFeatures(convert_categorical(self.discriminator.protected_variable,
                                                                                  self.hyper_parameters['category_maps']
                                                                                  ))

            protected_output = tf.gather(protected_feature(input_features), 0, axis=1)
            # calculating the loss of the discriminator
            disc_loss = self.loss(protected_output, discriminator_ouput)

            # calculate and apply the gradient of parameters of the discriminator network
            gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                            self.discriminator.model.trainable_variables)
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                             self.discriminator.model.trainable_variables))

            # calculate gradients of parameters of predictor network based on
            # loss in the discriminator network
            gradients_of_predictor_disc_loss = disc_tape.gradient(disc_loss, self.predictor.trainable_variables)
            # loss in the predictor network
            gradients_of_predictor_pred_loss = predictor_tape.gradient(predictor_loss, self.predictor.trainable_variables)

            gradients_of_predictor = self.__predictor_gradient(gradients_of_predictor_pred_loss,
                                                               gradients_of_predictor_disc_loss)

            # apply gradient updates
            self.predictor_optimizer.apply_gradients(zip(gradients_of_predictor, self.predictor.trainable_variables))

        return tf.cast(tf.greater(predictor_output, 0.0), dtype=tf.int32), predictor_loss
