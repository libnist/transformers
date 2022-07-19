# Import libraries
from tensorflow.keras import layers

from .embeddings import EmbeddingLayer
from .encoder_decoder_layers import *
from .encoder_decoder_layers import InverseFnetEncoderLayer


class VanillaEncoder(layers.Layer):
    def __init__(
            self, *, num_layers: int, d_model: int, num_heads: int,
            sequence_length: int, vocab_size: int, type_size: int,
            dense_dim: int = 1024, rate: float = .1,
            name: str = "VanillaEncoder", **kwargs
    ) -> layers.Layer:
        super(VanillaEncoder, self).__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.dense_dim = dense_dim
        self.rate = rate

        self.embedding = EmbeddingLayer(sequence_length=sequence_length,
                                        vocab_size=vocab_size,
                                        type_size=type_size,
                                        d_model=d_model,
                                        rate=rate)

        self.layers = [VanillaEncoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dense_dim=dense_dim,
                                           rate=rate)
                       for _ in range(num_layers)]

    def call(self, inputs, training: bool = False,
             with_embeddings: bool = False, padding_mask=None):
        outputs = self.embedding(inputs=inputs, training=training)
        embeddings = outputs

        for encoder in self.layers:
            outputs = encoder(inputs=outputs,
                              training=training,
                              padding_mask=padding_mask)
        if with_embeddings:
            return outputs, embeddings
        else:
            return outputs

    def get_config(self):
        config = super(VanillaEncoder, self).get_config()
        config.update({"num_layers": self.num_layers,
                       "d_model": self.d_model,
                       "num_heads": self.num_heads,
                       "sequence_length": self.sequence_length,
                       "vocab_size": self.vocab_size,
                       "type_size": self.type_size,
                       "dense_dim": self.dense_dim,
                       "rate": self.rate})
        return config


class VanillaDecoder(layers.Layer):
    def __init__(
            self, *, num_layers: int, d_model: int, num_heads: int,
            sequence_length: int, vocab_size: int, type_size: int,
            dense_dim: int = 1024, rate: float = .1,
            name: str = "VanillaDecoder", **kwargs
    ):
        super(VanillaDecoder, self).__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.dense_dim = dense_dim
        self.rate = rate

        self.embedding = EmbeddingLayer(sequence_length=sequence_length,
                                        vocab_size=vocab_size,
                                        type_size=type_size,
                                        d_model=d_model,
                                        rate=rate)

        self.layers = [VanillaDecoderLayer(d_model=d_model,
                                           num_heads=num_heads,
                                           dense_dim=dense_dim,
                                           rate=rate)
                       for _ in range(num_layers)]

    def call(self, inputs, enc_outputs, training: bool = False,
             look_ahead_mask=None, padding_mask=None):
        output = self.embedding(inputs, training=training)

        for decoder in self.layers:
            output = decoder(inputs=output, enc_outputs=enc_outputs,
                             training=training, padding_mask=padding_mask,
                             look_ahead_mask=look_ahead_mask)
        return output

    def get_config(self):
        config = super(VanillaDecoder, self).get_config()
        config.update({"num_layers": self.num_layers,
                       "d_model": self.d_model,
                       "num_heads": self.num_heads,
                       "sequence_length": self.sequence_length,
                       "vocab_size": self.vocab_size,
                       "type_size": self.type_size,
                       "dense_dim": self.dense_dim,
                       "rate": self.rate})
        return config


class FnetEncoder(layers.Layer):
    def __init__(
        self, *, num_layers: int, d_model: int, sequence_length: int,
        vocab_size: int, type_size: int, dense_dim: int = 1024,
        with_dense: bool = False, rate: float = .1,
        name: str = "FnetEncoder", **kwargs
    ):
        super(FnetEncoder, self).__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.dense_dim = dense_dim
        self.with_dense = with_dense
        self.rate = rate

        self.embedding = EmbeddingLayer(sequence_length=sequence_length,
                                        vocab_size=vocab_size,
                                        type_size=type_size,
                                        d_model=d_model,
                                        rate=rate)

        self.layers = [FnetEncoderLayer(d_model=d_model,
                                        dense_dim=dense_dim,
                                        with_dense=with_dense,
                                        rate=rate)
                       for _ in range(num_layers)]

    def call(self, inputs, training: bool = False,
             with_embeddings: bool = False):
        outputs = self.embedding(inputs=inputs, training=training)
        embeddings = outputs

        for encoder in self.layers:
            outputs = encoder(inputs=outputs,
                              training=training)
        if with_embeddings:
            return outputs, embeddings
        else:
            return outputs

    def get_config(self):
        config = super(FnetEncoder, self).get_config()
        config.update({"num_layers": self.num_layers,
                       "d_model": self.d_model,
                       "sequence_length": self.sequence_length,
                       "vocab_size": self.vocab_size,
                       "type_size": self.type_size,
                       "dense_dim": self.dense_dim,
                       "with_dense": self.with_dense,
                       "rate": self.rate})
        return config


class InverseFnetEncoder(layers.Layer):
    def __init__(
        self, *, num_layers: int, d_model: int, dense_dim: int = 1024,
        with_dense: bool = False, rate: float = .1,
        name: str = "FnetEncoder", **kwargs
    ):
        super(InverseFnetEncoder, self).__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.dense_dim = dense_dim
        self.with_dense = with_dense
        self.rate = rate

        self.layers = [InverseFnetEncoderLayer(d_model=d_model,
                                               dense_dim=dense_dim,
                                               with_dense=with_dense,
                                               rate=rate)
                       for _ in range(num_layers)]

    def call(self, inputs, training: bool = False,
             with_embeddings: bool = False):
        outputs = inputs

        for encoder in self.layers:
            outputs = encoder(inputs=outputs,
                              training=training)
        return outputs

    def get_config(self):
        config = super(InverseFnetEncoder, self).get_config()
        config.update({"num_layers": self.num_layers,
                       "d_model": self.d_model,
                       "dense_dim": self.dense_dim,
                       "with_dense": self.with_dense,
                       "rate": self.rate})
        return config
