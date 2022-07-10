# Import libraries
import tensorflow as tf
from keras import layers

# __all__ defines what can be accessed from outside
__all__ = ["FnetAttention"]


class FnetAttention(layers.Layer):
    """
    This is the f-net self-attention mechanism introduced in a article by the 
    same name. This intra attention mechanism uses a fourier transform and
    avoids any dot products, as a result it's faster than MultiHeadAttention.
    You can decide whether you wan't a Dense layer after fourier transform by
    the `with_dense` parameter, if so the `rate` will be counted as the dropout
    rate contigous to the Dense, and `d_model` will be the output dim of that
    Dense layer.
    rate: [0, 1]
    """

    def __init__(
        self, *, with_dense: bool = False, d_model: int = 512,
        rate: float = .1, name: str = "FnetAttention", **kwargs
    ) -> layers.Layer:
        super(FnetAttention, self).__init__(name=name, **kwargs)

        self.with_dense = with_dense
        self.d_model = d_model
        self.rate = rate

        if with_dense:
            self.dense = layers.Dense(units=d_model)
            self.dropout = layers.Dropout(rate=rate)

        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training: bool = True):
        """
        inputs: an already embedded tensor of inputs in the sahpe:
        (batch_size, seq_len, embedding_dim)
        """
        outputs = tf.cast(inputs, dtype=tf.complex64)
        outputs = tf.math.real(tf.signal.fft2d(outputs))

        if self.with_dense:
            outputs = self.dense(outputs)
            outputs = self.dropout(outputs, training=training)

        outputs = self.layernorm(inputs + outputs)

        return outputs

    def get_config(self):
        config = super(FnetAttention, self).get_config()
        config.update({"with_dense": self.with_dense,
                       "d_model": self.d_model,
                       "rate": self.rate})
        return config


class InverseFnetAttention(layers.Layer):
    """
    This is the Inverse f-net self-attention mechanism.This intra attention 
    mechanism uses a inverse fourier transform, this attention mechanism is 
    built in case of creating a autoencoder based on transformers. You can 
    decide whether you wan't a Dense layer after reverse fourier transform by
    the `with_dense` parameter, if so the `rate` will be counted as the dropout
    rate contigous to the Dense, and `d_model` will be the output dim of that
    Dense layer.
    rate: [0, 1]
    """

    def __init__(
        self, *, with_dense: bool = False, d_model: int = 512,
        rate: float = .1, name: str = "ReverseFnetAttention", **kwargs
    ) -> layers.Layer:
        super(InverseFnetAttention, self).__init__(name=name, **kwargs)

        self.with_dense = with_dense
        self.d_model = d_model
        self.rate = rate

        if with_dense:
            self.dense = layers.Dense(units=d_model)
            self.dropout = layers.Dropout(rate=rate)

        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training: bool = True):
        """
        inputs: an already embedded tensor of inputs in the sahpe:
        (batch_size, seq_len, embedding_dim)
        """
        
        outputs = tf.cast(inputs, dtype=tf.complex64)
        outputs = tf.math.real(tf.signal.ifft2d(outputs))

        if self.with_dense:
            outputs = self.dense(outputs)
            outputs = self.dropout(outputs, training=training)

        outputs = self.layernorm(inputs + outputs)

        return outputs

    def get_config(self):
        config = super(FnetAttention, self).get_config()
        config.update({"with_dense": self.with_dense,
                       "d_model": self.d_model,
                       "rate": self.rate})
        return config
