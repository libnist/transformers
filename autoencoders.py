# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.embedding import EmbeddingLayer
from utils.encoder_decoder import FnetEncoderLayer, InverseFnetEncoderLayer
from masters.encoder_decoder import Encoder, InverseEncoder


class FnetAutoEncoder(keras.Model):
    """
    An AutoEncoder based on the idea of transformers in order to compress
    long text sequences.
    """

    def __init__(
        self, *, d_model: int, number_of_layers: int, sequence_len: int,
        vocab_size: int, type_size: int, dense_dim: int = 1024,
        with_dense: bool = False, rate: float = .1,
        name: str = "FnetAutoEncoder"
    ) -> keras.Model:
        super(FnetAutoEncoder, self).__init__(name=name)

        self.d_model = d_model
        self.number_of_layers = number_of_layers
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.dense_dim = dense_dim
        self.with_dense = with_dense
        self.rate = rate

        # Encoder
        embedding = EmbeddingLayer(sequence_length=sequence_len,
                                   vocab_size=vocab_size,
                                   type_size=type_size,
                                   d_model=d_model,
                                   rate=rate)

        fnet_encoder = FnetEncoderLayer(d_model=d_model,
                                        dense_dim=dense_dim,
                                        with_dense=with_dense,
                                        rate=rate)

        self.encoder = Encoder(layer=fnet_encoder,
                               number_of_layers=number_of_layers,
                               embedding_layer=embedding,
                               name="FnetEncoder")

        # AutoEncoder

        # InverseEncoder
        inv_fnet_encoder = InverseFnetEncoderLayer(d_model=d_model,
                                                   dense_dim=dense_dim,
                                                   with_dense=with_dense,
                                                   rate=rate)

        self.inv_encoder = InverseEncoder(layer=inv_fnet_encoder,
                                          number_of_layers=number_of_layers,
                                          embedding_layer=None,
                                          name="InverseFnetEncoder")

    def call(self, inputs, training: bool = False, **kwargs):
        pass

    def get_config(self):
        pass
