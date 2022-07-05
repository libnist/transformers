# Import libraries
from keras import layers

# __all__ defines the thing that can be accessed from outside
__all__ = ["PFFN"]


class PFFN(layers.Layer):
    """
    PFFN is a feed forward neural network with one dense layer in `dense_dim`
    dimensions, and another with `d_model` dimensions. `rate` is used for the
    dropout layer after the second Dense layer.
    """

    def __init__(
        self, *, d_model: int, dense_dim: int,
        rate: float = .1, name: str = "PFFN", **kwargs
    ) -> layers.Layer:
        super(PFFN, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.dense_dim = dense_dim
        self.rate = rate

        self.dense_1 = layers.Dense(units=dense_dim, activation="relu")
        self.dense_2 = layers.Dense(units=d_model)
        self.dropout = layers.Dropout(rate=rate)
        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training: bool = True):
        """
        inputs have to be already embedded and in shape:
        (batch_size, seq_len, d_model)
        """
        outputs = self.dense_1(inputs)  # (batch_size, seq_len, dense_dim)
        outputs = self.dense_2(outputs)  # (batch_size, seq_len, d_model)
        outputs = self.dropout(outputs, training=training)
        outputs = self.layernorm(inputs + outputs)

        return outputs

    def get_config(self):
        config = super(PFFN, self).get_config()
        config.update({"d_model": self.d_model,
                       "dense_dim": self.dense_dim,
                       "rate": self.rate})
        return config
