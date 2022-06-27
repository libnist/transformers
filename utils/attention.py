# Import libraries
import tensorflow as tf
from keras import layers

__all__ = ["FnetAttention"]

# The attention below was introduced in
# F-net article


class FnetAttention(layers.Layer):
    def __init__(self, *, with_dense=False, d_model=512,
                 rate=.1, name="FnetAttention", **kwargs):
        super(FnetAttention, self).__init__(name=name, **kwargs)

        self.with_dense = with_dense
        self.d_model = d_model
        self.rate = rate

        if with_dense:
            self.dense = layers.Dense(units=d_model)
            self.dropout = layers.Dropout(rate=rate)

        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training=True):
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
