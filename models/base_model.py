import tensorflow as tf
from tqdm import tqdm


class BaseModel:

    def __init__(self):
        pass

    def __train_loop(self, train_dataset):
        """
        Execute one epoch of training
        :param train_dataset: the entire training dataset
        :return:
        """

        # reset the state of all metrics
        [metric.reset_states() for metric in self.metrics]

        # iterate over the training dataset
        for feature_batch, labels in tqdm(train_dataset):
            # execute single trainning step
            predictor_output, predictor_loss = self._train_step(feature_batch, labels)

            # calculate metrics
            self.metrics[0](predictor_loss)
            [metric(labels, predictor_output) for metric in self.metrics[1:]]

    def train(self, train_dataset, epochs, validation_dataset=None):
        """
        Execute entire training
        :param train_dataset: Training dataset
        :param epochs: number of epochs
        :param validation_dataset: Validation dataset to track model performance
        :return:
        """

        # iterate over epochs
        for epoch in range(epochs):

            print(f"Epoch: {epoch}")

            # run the training loop
            print("Training:")
            self.__train_loop(train_dataset)

            # log the metrics
            train_metric_str = ", ".join([f"{type(metric).__name__}: {metric.result()}"
                                          for metric in self.metrics])
            print(train_metric_str)
            print("------------------------------------------------------\n")

            # if validation set is given run the evaluation over it
            if validation_dataset is not None:
                print("Validation:")
                self.evaluate(validation_dataset)

    @tf.function
    def __predict_batch(self, feature_batch, from_logits=False):
        """
        Predict labels from one batch of the data
        :param feature_batch: input features
        :param from_logits:
        :return:
        """
        # output of the network: here depends on the final activation function of the last layer
        prediction = self.predictor(feature_batch, training=False)

        # if no activation was given
        if from_logits:
            prediction = tf.sigmoid(prediction)

        return prediction

    def evaluate(self, dataset):
        """
        Evaluates the model over given dataset
        :param dataset: (input_features, labels)
        :return:
        """
        # reset the metrics
        [metric.reset_states() for metric in self.metrics]

        # iterate over a dataset
        for feature_batch, labels in tqdm(dataset):

            # predict the output
            prediction = self.__predict_batch(feature_batch,  True)
            [metric(labels, prediction) for metric in self.metrics]

        # evaluate metrics
        metric_results = [metric.result() for metric in self.metrics]
        metric_str = ", ".join([f"{type(metric).__name__}: {metric_results[index]}"
                                for index, metric in enumerate(self.metrics)])
        print(metric_str)
        print("------------------------------------------------------\n")

        return metric_results

    def _base_predict(self, dataset):
        """
       Predicts probability of labels for given dataset
       :param dataset: (input_features, labels) !!!! Cause used the same pipeline (lazy). other
       scenarios fill with dummy column
       :return: TF tensor
       """
        output = None
        index = 0

        # iterate over a dataset
        for feature_batch, labels in tqdm(dataset):
            # predict the output
            prediction = self.__predict_batch(feature_batch, True)
            # collect in the output
            if index == 0:
                output = prediction
            else:
                output = tf.concat([output, prediction], axis=0)
            index += 1

        return output

    def predict_proba(self, dataset):
        """
        Predicts probability of labels for given dataset
        :param dataset: (input_features, labels) !!!! Cause used the same pipeline (lazy). other
        :param dataset:
        :return: numpy array
        """
        output = self._base_predict(dataset)
        return output.numpy()

    def predict(self, dataset):
        """
        Predicts probability of labels for given dataset
        :param dataset: (input_features, labels) !!!! Cause used the same pipeline (lazy). other
        :param dataset:
        :return: numpy array
        """
        output = self._base_predict(dataset)
        output = tf.cast(tf.greater(output, 0.5), dtype=tf.int32)
        return output.numpy()
