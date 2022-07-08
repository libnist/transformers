# Import Libraries
from keras import layers

from .attention import FnetAttention
from .feedforward import PFFN

# __all__ defines what can be accessed from outside
__all__ = ["VanillaEncoderLayer",
           "FnetEncoderLayer",
           "VanillaDecoderLayer"]


class VanillaEncoderLayer(layers.Layer):
    """
    A single layer encoder w/ MHA.
    d_model: used to define d_model of nested layers like MHA and PFFN and the 
    overal output shape.
    numn_heads: number of heads in MHA layer.
    dense_dim: passed to PFFN.
    rate: each of nested layers have Dropout layers, rate defines the droput
    rate for those [0, 1].
    name: name of the layer.
    """

    def __init__(
        self, *, d_model: int, num_heads: int, dense_dim: int = 1024,
        rate: float = .1, name: str = "VanillaEncoderLayer", **kwargs
    ) -> layers.Layer:
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

    def call(self, inputs, training: bool = False, padding_mask=None):
        """
        inputs are in shape: (batch_size, seq_len, d_model) and already
        embedded by an embeddingLayer.
        """
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


class FnetEncoderLayer(layers.Layer):
    """
    A single encoder layer that uses fft rather MHA.
    d_model: used to define d_model of nested layers like PFFN, and the overall
    outputs shape.
    wiht_dense: whether fnet-attention should use dense layers.
    dense_dim: passed to PFFN.
    rate: each of nested layers have Dropout layers, rate defines the droput
    rate for those [0, 1].
    name: name of the layer.
    """

    def __init__(
        self, *, d_model: int, dense_dim: int = 1024, with_dense: bool = False,
        rate: float = .1, name: str = "FnetEncoderLayer", **kwargs
    ) -> layers.Layer:
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

    def call(self, inputs, training: bool = False, **kwargs):
        """
        inputs are already embedded tokens in the shape: 
        (batch_size, seq_len, d_model)
        """
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


class VanillaDecoderLayer(layers.Layer):
    """
    A single decoder layer proposed in AIYN paper.
    d_model: used to define d_model of nested layers like MHA and PFFN and the 
    overall output shape.
    numn_heads: number of heads in MHA layer.
    dense_dim: passed to PFFN.
    rate: each of nested layers have Dropout layers, rate defines the droput
    rate for those [0, 1].
    name: name of the layer.
    """

    def __init__(
        self, *, d_model: int, num_heads: int, dense_dim: int = 1024,
        rate: int = .1, name: str = "VanillaDecoder", **kwargs
    ) -> layers.Layer:
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

    def call(
        self, inputs, enc_outputs, training: bool = False,
        padding_mask=None, look_ahead_mask=None
    ):
        """
        inputs are already embedded and in shape: 
        (batch_size, seq_len, d_mdoel)
        enc_outputs comes from the last layer of Encoder.
        padding mask is used in the second MHA while lookahead mask in the 
        first.
        """
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
