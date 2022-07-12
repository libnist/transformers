# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .utils.embedding import EmbeddingLayer
from .utils.encoder_decoder import FnetEncoderLayer, InverseFnetEncoderLayer
from .masters.encoder_decoder import Encoder, InverseEncoder


class FnetAutoEncoder(keras.Model):
    """
    An AutoEncoder based on the idea of transformers in order to compress
    long text sequences.
    """

    def __init__(
        self, *, d_model: int, number_of_layers: int, sequence_len: int,
        vocab_size: int, type_size: int, dense_layers: list, latent_dim: int,
        maxlen_pad: int = 1024, dense_dim: int = 1024,
        with_dense: bool = False, rate: float = .1,
        name: str = "FnetAutoEncoder"
    ) -> keras.Model:
        super(FnetAutoEncoder, self).__init__(name=name)

        self.d_model = d_model
        self.number_of_layers = number_of_layers
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.dense_layers = dense_layers
        self.latent_dim = latent_dim
        self.maxlen_pad = maxlen_pad
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

        # ML
        encoder_outputs = keras.Input(shape=(maxlen_pad, d_model),
                                      dtype=tf.float64)
        outputs = layers.Dense(
            units=vocab_size, activation="softmax"
        )(encoder_outputs)
        self.ml_model = keras.Model(inputs=encoder_outputs,
                                    outputs=outputs,
                                    name="Masked Language Model")

        # AutoEncoder
        encoder_outputs = keras.Input(shape=(maxlen_pad, d_model),
                                      dtype=tf.float64)

        x = layers.Permute((2, 1))(encoder_outputs)
        x = layers.Dense(units=maxlen_pad, activation="relu")(x)
        x = layers.Dropout(rate=rate)(x)

        for units in dense_layers:
            x = layers.Dense(units=units, activation="relu")(x)
            x = layers.Dropout(rate=rate)(x)

        z_mean = layers.Dense(units=latent_dim, name="z_mean")(x)
        z_var = layers.Dense(units=latent_dim, name="z_var")(x)
        z = Sampling()([z_mean, z_var])
        z = layers.Permute((2, 1))(z)

        x = layers.Permute((2, 1))(z)

        for units in dense_layers[::-1]:
            x = layers.Dense(units=units, activation="relu")(x)
            x = layers.Dropout(rate=rate)(x)

        x = layers.Dense(units=maxlen_pad, activation="relu")(x)
        x = layers.Dropout(rate=rate)(x)
        outputs = layers.Permute((2, 1))(x)

        self.vae = keras.Model(inputs=encoder_outputs,
                               outputs={"latent": z,
                                        "output": outputs},
                               name="VAE")

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
        tokens, types = inputs
        tokens = keras.utils.pad_sequences(
            tokens, padding="post",
            truncating="post", maxlen=self.maxlen_pad
        )
        types = keras.utils.pad_sequences(
            types, value=tf.reduce_max(types),
            padding="post", truncating="post", maxlen=self.maxlen_pad
        )
        inputs = (tokens, types)
        outputs = self.encoder(inputs, training=training)
        vae_dict = self.vae(outputs, training=training)
        outputs = self.inv_encoder(vae_dict["output"], training=training)
        return outputs, vae_dict["latent"]

    def train_step(self, inputs):
        pass

    def get_config(self):
        pass


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_var) * epsilon
