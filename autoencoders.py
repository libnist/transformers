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
        inputs = keras.Input(shape=(maxlen_pad, d_model),
                             dtype=tf.int64)
        outputs = layers.Dense(units=vocab_size)(inputs)
        self.ml_model = keras.Model(inputs=inputs,
                                    outputs=outputs,
                                    name="MaskedLanguageModel")

        # VAutoEncoder
        vae_inputs = keras.Input(shape=(maxlen_pad, d_model),
                                 dtype=tf.float64)

        x = layers.Permute((2, 1))(vae_inputs)
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

        self.vae = keras.Model(inputs=vae_inputs,
                               outputs={"latent": z,
                                        "z_mean": z_mean,
                                        "z_var": z_var,
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

    def call(self, inputs, training: bool = False, ** kwargs):
        enc_outputs, embeddings = self.encoder(inputs,
                                               training=training,
                                               with_embeddings=True)

        vae_dict = self.vae(enc_outputs, training=training)
        inv_enc_outputs = self.inv_encoder(
            vae_dict["output"],
            training=training
        )
        kl_loss = (-0.5 * (1 + vae_dict["z_var"]
                          - tf.square(vae_dict["z_mean"]) -
                          tf.exp(vae_dict["z_var"])))
        self.add_loss(tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))
        return {"embeddings": embeddings,
                "outputs": inv_enc_outputs,
                "latent": vae_dict["latent"],
                "enc_outputs": enc_outputs}

    def compile(self, ml_optimizer, ae_optimizer,
                ml_loss, **kwargs):
        super(FnetAutoEncoder, self).compile(**kwargs)
        self.ml_optimizer = ml_optimizer
        self.ae_optimizer = ae_optimizer
        self.ml_loss = ml_loss

    def train_step(self, inputs):
        tokens, types = inputs
        tf.print("token_shape", tf.shape(tokens))

        # Get the trainable variable of each part
        ml_vars = (self.encoder.trainable_variables +
                   self.ml_model.trainable_variables)
        vae_vars = (self.encoder.trainable_variables +
                    self.vae.trainable_variables +
                    self.inv_encoder.trainable_variables)

        # Train ml_model
        with tf.GradientTape() as tape:
            latent = self(inputs, training=True)["enc_outputs"]
            predictions = self.ml_model(latent, training=True)
            tf.print("preds_shape", tf.shape(predictions))
            ml_loss = self.ml_loss(y_true=tokens,
                                   y_pred=predictions)

        grads = tape.gradient(ml_loss, ml_vars)
        self.ml_optimizer.apply_gradients(zip(grads, ml_vars))

        # Train vae
        with tf.GradientTape() as tape:
            vae_back = self(inputs, training=True)
            embeddings = vae_back["embeddings"]
            outputs = vae_back["outputs"]
            ae_loss = self.compiled_loss(y_true=embeddings,
                                         y_pred=outputs)
            ae_loss += self.losses

        grads = tape.gradient(ae_loss, vae_vars)
        self.ae_optimizer.apply_gradients(zip(grads, vae_vars))

        return {"MLM_Loss": ml_loss,
                "VAE_Loss": ae_loss}

    def get_config(self):
        pass


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_var) * epsilon
