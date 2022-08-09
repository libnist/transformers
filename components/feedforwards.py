# Import libraries
from audioop import ratecv
from keras import layers

# __all__ defines the thing that can be accessed from outside
__all__ = ["FFNN",
           "CNN"]


class FFNN(layers.Layer):
    """
    FFNN is a feed forward neural network with one dense layer in `dense_dim`
    dimensions, and another with `d_model` dimensions. `rate` is used for the
    dropout layer after the second Dense layer.
    """

    def __init__(
        self, *, d_model: int, dense_dim: int,
        rate: float = .1, name: str = "FFNN", **kwargs
    ) -> layers.Layer:
        super(FFNN, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.dense_dim = dense_dim
        self.rate = rate

        self.dense_1 = layers.Dense(units=dense_dim, activation="relu")
        self.dense_2 = layers.Dense(units=d_model)
        self.dropout = layers.Dropout(rate=rate)
        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training: bool = False):
        """
        inputs have to be already embedded and in shape:
        (batch_size, seq_len, d_model)
        """
        outputs = self.dense_1(
            inputs, training=training)  # (batch_size, seq_len, dense_dim)
        # (batch_size, seq_len, d_model)
        outputs = self.dense_2(outputs, training=training)
        outputs = self.dropout(outputs, training=training)
        outputs = self.layernorm(inputs + outputs, training=training)

        return outputs

    def get_config(self):
        config = super(FFNN, self).get_config()
        config.update({"d_model": self.d_model,
                       "dense_dim": self.dense_dim,
                       "rate": self.rate})
        return config


class CNN(layers.Layer):

    def __init__(self, *, dim_filters, rate=.1, name="CNN"):
        super(CNN, self).__init__(name=name)

        self.dim_filters = dim_filters
        self.rate = rate

        # self.conv1 = layers.Conv1D(expanded_filters,
        #                            3, activation="relu",
        #                            padding="same")
        self.conv2 = layers.Conv1D(dim_filters,
                                   3, activation="relu",
                                   padding="same")
        self.dropout = layers.Dropout(rate)
        self.normalize = layers.LayerNormalization()

    def call(self, inputs):
        # outputs = self.conv1(inputs)
        outputs = self.conv2(inputs)
        outputs = self.dropout(outputs)
        return self.normalize(inputs + outputs)

    def get_config(self):
        config = super(CNN, self).get_config()
        config.update({"dim_filters": self.dim_filters,
                       "rate": self.rate})
        return config
