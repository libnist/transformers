# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .components.embeddings import EmbeddingLayer
from .components.encoder_decoder_layers import FnetEncoderLayer, InverseFnetEncoderLayer
from .masters.encoder_decoder import Encoder, InverseEncoder


class FnetAutoEncoder(keras.Model):
    """
    An AutoEncoder based on the idea of transformers in order to compress
    long text sequences.
    """

    def __init__(
        self, *, d_model: int, number_of_layers: int, sequence_len: int,
        vocab_size: int, type_size: int, dense_layers: list, latent_dim: int,
        dense_dim: int = 1024, with_dense: bool = False, rate: float = .1,
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
        self.dense_dim = dense_dim
        self.with_dense = with_dense
        self.rate = rate

        # Encoder Layer-start
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
        # Encoder Layer-end

        # Masked Language Model-start
        self.mlm_model = keras.Sequential(
            [keras.Input(shape=(sequence_len, d_model)),
             layers.Dense(units=vocab_size)],
            name="MaskedLanguageModel"
        )
        # Masked Language Model-end

        # Variational Auto Encoder-start
        # Variational Auto Encoder->encoder-start
        vae_encoder_inputs = keras.Input(shape=(sequence_len, d_model),
                                         dtype=tf.float32)
        x = layers.Permute((2, 1))(vae_encoder_inputs)
        x = layers.Dense(units=sequence_len, activation="relu")(x)
        x = layers.Dropout(rate=rate)(x)

        for units in dense_layers:
            x = layers.Dense(units=units, activation="relu")(x)
            x = layers.Dropout(rate=rate)(x)

        z_mean = layers.Dense(units=latent_dim, name="z_mean")(x)
        z_var = layers.Dense(units=latent_dim, name="z_var")(x)
        z = Sampling()([z_mean, z_var])
        vae_encoder_outputs = layers.Permute((2, 1))(z)
        self.vae_encoder = keras.Model(inputs=vae_encoder_inputs,
                                       outputs={"z_mean": z_mean,
                                                "z_var": z_var,
                                                "latent": vae_encoder_outputs},
                                       name="VAEEncoder")
        # Variational Auto Encoder->encoder-end

        # Variational Auto Encoder->decoder-start
        vae_decoder_inputs = keras.Input(shape=(latent_dim, d_model),
                                         dtype=tf.float32)
        x = layers.Permute((2, 1))(vae_decoder_inputs)

        for units in dense_layers[::-1]:
            x = layers.Dense(units=units, activation="relu")(x)
            x = layers.Dropout(rate=rate)(x)

        x = layers.Dense(units=sequence_len, activation="relu")(x)
        x = layers.Dropout(rate=rate)(x)
        vae_decoder_outputs = layers.Permute((2, 1))(x)

        self.vae_decoder = keras.Model(inputs=vae_decoder_inputs,
                                       outputs=vae_decoder_outputs,
                                       name="VAEDecoder")
        # Variational Auto Encoder->decoder-end
        # Variational Auto Encoder-end

        # Inverse Encoder-start
        inv_fnet_encoder = InverseFnetEncoderLayer(d_model=d_model,
                                                   dense_dim=dense_dim,
                                                   with_dense=with_dense,
                                                   rate=rate)

        self.inv_encoder = InverseEncoder(layer=inv_fnet_encoder,
                                          number_of_layers=number_of_layers,
                                          embedding_layer=None,
                                          name="InverseFnetEncoder")
        # Inverse Encoder-end

        self.build(None)

    def build(self, input_shape):
        input = tf.zeros(shape=(1, self.sequence_len), dtype=tf.int32)
        input = (input, input)
        enc_out = self.encoder(input)
        inv_enc_out = self.inv_encoder(enc_out)

    def call(self, inputs, training: bool = False, ** kwargs):
        outputs = self.encoder(inputs, training=training)
        outputs = self.vae_encoder(outputs, training=training)
        return outputs["latent"]

    def compile(self, mlm_optimizer, vae_optimizer,
                mlm_loss, vae_loss, **kwargs):
        super(FnetAutoEncoder, self).compile(**kwargs)
        self.mlm_optimizer = mlm_optimizer
        self.vae_optimizer = vae_optimizer
        self.mlm_loss = mlm_loss
        self.vae_loss = vae_loss

        self.mlm_loss_metric = keras.metrics.Mean()
        self.vae_loss_metric = keras.metrics.Mean()

    def mlm_phase(self, inputs):
        outputs = self.encoder(inputs, training=True)
        return self.mlm_model(outputs, training=True)

    def kl_loss(self, z_mean, z_var):
        kl_loss = (-0.5 * (1 + z_var - tf.square(z_mean) - tf.exp(z_var)))
        return tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

    def vae_phase(self, inputs):
        outputs, embeddings = self.encoder(inputs,
                                           training=True,
                                           with_embeddings=True)
        outputs = self.vae_encoder(outputs, training=True)
        z_mean, z_var, latent = (outputs["z_mean"],
                                 outputs["z_var"],
                                 outputs["latent"])
        outputs = self.vae_decoder(latent, training=True)
        outputs = self.inv_encoder(outputs, training=True)
        return outputs, embeddings, self.kl_loss(z_mean, z_var)

    def train_step(self, inputs):
        tokens, _ = inputs

        # Get the trainable variable of each part
        mlm_vars = (self.encoder.trainable_variables +
                    self.mlm_model.trainable_variables)
        vae_vars = (self.encoder.trainable_variables +
                    self.vae_encoder.trainable_variables +
                    self.vae_decoder.trainable_variables +
                    self.inv_encoder.trainable_variables)

        # Train mlm_model
        with tf.GradientTape() as tape:
            predictions = self.mlm_phase(inputs)
            mlm_loss = self.mlm_loss(y_true=tokens,
                                     y_pred=predictions)

        grads = tape.gradient(mlm_loss, mlm_vars)
        self.mlm_optimizer.apply_gradients(zip(grads, mlm_vars))

        # Train vae
        with tf.GradientTape() as tape:
            predictions, embeddings, kl_loss = self.vae_phase(inputs)
            vae_loss = self.vae_loss(y_true=embeddings,
                                     y_pred=predictions)
            vae_loss += kl_loss

        grads = tape.gradient(vae_loss, vae_vars)
        self.vae_optimizer.apply_gradients(zip(grads, vae_vars))

        self.mlm_loss_metric.update_state(mlm_loss)
        self.vae_loss_metric.update_state(vae_loss)

        return {"MLM_Loss": self.mlm_loss_metric.result(),
                "VAE_Loss": self.vae_loss_metric.result()}

    def get_config(self):
        config = super(FnetAutoEncoder, self).get_config()
        config.update({"d_model": self.d_model,
                       "number_of_layers": self.number_of_layers,
                       "sequence_len": self.sequence_len,
                       "vocab_size": self.vocab_size,
                       "type_size": self.type_size,
                       "dense_layers": self.dense_layers,
                       "latent_dim": self.latent_dim,
                       "dense_dim": self.dense_dim,
                       "with_dense": self.with_dense,
                       "rate": self.rate})
        return config

    @property
    def metrics(self):
        return [self.mlm_loss_metric,
                self.vae_loss_metric]


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_var) * epsilon
