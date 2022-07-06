# Imort libraries
import tensorflow as tf
import keras

# __all__ defines what can be accessed from outside
__all__ = ["SCCELoss",
           "Accuracy"]


class SCCELoss(keras.losses.Loss):
    """
    This is a custom SparceCategoricalCrossentropy loss.
    """

    def __init__(self, name: str = "SCCE") -> keras.losses.Loss:
        super(SCCELoss, self).__init__(name=name)
        self.scce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none"
        )

    def call(self, y_ture, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_ture, 0))
        loss_ = self.scce(y_ture, y_pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        loss = tf.reduce_sum(loss_)/tf.reduce_sum(mask)
        return loss


class Accuracy(keras.metrics.Metric):
    """
    This is a custom Accuracy for transformer models.
    """
    def __init__(self, name: str = "Accuracy") -> keras.metrics.Metric:
        super(Accuracy, self).__init__(name=name)
        self.count = tf.Variable(0, dtype=tf.float32)
        self.acc = tf.Variable(0, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        accuracies = tf.equal(y_true, tf.argmax(y_pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        accuracy = tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

        self.count.assign_add(1.)
        self.acc.assign_add(accuracy)

    def result(self):
        return self.acc / self.count

    def reset_state(self):
        self.count.assign(0.)
        self.acc.assign(0.)
