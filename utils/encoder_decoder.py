# Import Libraries
from keras import layers

from .attention import FnetAttention
from .feedforward import PFFN

__all__ = ["VanillaEncoderLayer",
           "FnetEncoderLayer",
           "VanillaDecoderLayer"]

# Vanilla EncoderLayer


class VanillaEncoderLayer(layers.Layer):
    def __init__(self, *, d_model, num_heads, dense_dim=1024,
                 rate=.1, name="VanillaEncoderLayer", **kwargs):
        super(VanillaEncoderLayer, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.rate = rate

        self.mha = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=d_model,
                                             dropout=rate)
        self.pffn = PFFN(d_model=d_model,
                         dense_dim=dense_dim,
                         rate=rate)

    def call(self, inputs, training=False, padding_mask=None):

        outputs = self.mha(query=inputs, key=inputs, value=inputs,
                           attention_mask=padding_mask, training=training)
        outputs = self.pffn(inputs=outputs, training=training)

        return outputs

    def get_config(self):
        config = super(VanillaEncoderLayer, self).get_config()
        config.update({"d_model": self.d_model,
                       "num_heads": self.num_heads,
                       "dense_dim": self.dense_dim,
                       "rate": self.rate})
        return config

# Fnet-EncoderLayer


class FnetEncoderLayer(layers.Layer):
    def __init__(self, *, d_model, dense_dim=1024, with_dense=False,
                 rate=.1, name="FnetEncoderLayer", **kwargs):
        super(FnetEncoderLayer, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.dense_dim = dense_dim
        self.with_dense = with_dense
        self.rate = rate

        self.fnet = FnetAttention(
            with_dense=with_dense,
            d_model=d_model,
            rate=rate
        )
        self.pffn = PFFN(
            d_model=d_model,
            dense_dim=dense_dim,
            rate=rate
        )

    def call(self, inputs, training=False, **kwargs):

        outputs = self.fnet(inputs=inputs, training=training)
        outputs = self.pffn(inputs=outputs, training=training)

        return outputs

    def get_config(self):
        config = super(FnetEncoderLayer, self).get_config()
        config.update({"d_model": self.d_model,
                       "dense_dim": self.dense_dim,
                       "with_dense": self.with_dense,
                       "rate": self.rate})
        return config

# Vanilla Decoder


class VanillaDecoderLayer(layers.Layer):
    def __init__(self, *, d_model, num_heads, dense_dim=1024,
                 rate=.1, name="VanillaDecoder", **kwargs):
        super(VanillaDecoderLayer, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.rate = rate

        self.mha1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=rate
        )
        self.mha2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=rate
        )
        self.pffn = PFFN(
            d_model=d_model,
            dense_dim=dense_dim,
            rate=rate
        )

        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs, enc_outputs, training=False,
             padding_mask=None, look_ahead_mask=None):

        outputs_mha1 = self.mha1(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=look_ahead_mask,
            training=training
        )
        outputs_mha1 = self.layernorm1(inputs + outputs_mha1)

        outputs_mha2 = self.mha2(
            query=outputs_mha1,
            key=enc_outputs,
            value=enc_outputs,
            attention_mask=padding_mask,
            training=training
        )
        outputs_mha2 = self.layernorm2(outputs_mha1 + outputs_mha2)

        outputs = self.pffn(inputs=outputs_mha2, training=training)

        return outputs

    def get_config(self):
        config = super(VanillaDecoderLayer, self).get_config()
        config.update({"d_model": self.d_model,
                       "num_heads": self.num_heads,
                       "dense_dim": self.dense_dim,
                       "rate": self.rate})
        return config
